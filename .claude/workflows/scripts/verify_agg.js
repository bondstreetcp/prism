export const meta = {
  name: 'verify-agg-report',
  description: 'Adversarially verify the aggregated Goldman+IBKR end-to-end report against the raw source files',
  phases: [
    { title: 'Verify', detail: 'one reconciler per dimension, recomputing from raw files' },
    { title: 'Audit', detail: 'independent auditor recomputes and tries to refute each verdict' },
  ],
}

const ART = String(args.art)
const GOLDMAN = String(args.goldman)
const IBKR = String(args.ibkr)

const COMMON = `
You are verifying an END-TO-END test of an aggregated multi-broker portfolio risk report.
Two broker files were consolidated into ONE book and a full risk pipeline was run.

Source files (raw, on disk):
  GOLDMAN = ${GOLDMAN}   (Goldman "Intraday Position" export, quantities only, account AWKF1209)
  IBKR    = ${IBKR}      (Interactive Brokers Activity Statement, account U12004488)

Dumped artifacts directory (ART): ${ART}
  parse_goldman.csv / parse_goldman_meta.json   individual Goldman parse
  parse_ibkr.csv / parse_ibkr_meta.json         individual IBKR parse
  parse_merged.csv / parse_merged_meta.json     the consolidated parse (merge_parse_results)
  positions.csv        position-level analytics (priced: mv, exposure, delta, sector, cap, region)
  issuers.csv          issuer-level (netted) analytics
  summary.json         the analytics summary dict (exposures, aum, cash, mv, splits, liquidity)
  headline.json  sector_table.json  cap_table.json  region_table.json
  factor_risk.json  scenarios.json  crowding.json  alert_hits.json  issues.json
  pdf_info.json        {path, exists, size_bytes, n_pages}
  pipeline_log.txt     the full run log

Aggregation contract (what the code SHOULD do):
- merge key: equities keyed "EQ:<underlying>", options keyed "OPT:<contract_key>".
  Positions on the SAME key sum their qty across files; a net-zero key is DROPPED.
- cash: sum of files that report cash (only IBKR does; Goldman has none) -> IBKR cash.
- nav: sum of file NAVs, BUT set to None if not ALL files report a NAV (Goldman has none) -> None.
- asof: max of the file as-of dates (Goldman 2026-07-07, IBKR 2026-07-10) -> 2026-07-10.
- accounts: sorted union. source label "goldman+ibkr".
- AUM: with nav None and no --cash, analytics computes aum = mv_net + cash.

RULES:
- RECOMPUTE from the raw CSVs and artifacts using python/pandas via the Bash tool. Do NOT trust
  any single artifact's claim — cross-check it. Run actual code; show the numbers.
- Use tolerances for floats (abs/rel 1e-6). Ints (counts, share qty) must match exactly.
- A "WARN" is for a real but expected/by-design limitation (e.g. Goldman cash unknown so AUM
  is understated). A "FAIL" is a genuine correctness defect. "PASS" is clean.
- Be concrete: every check gets expected vs actual.
`

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    dimension: { type: 'string' },
    verdict: { type: 'string', enum: ['PASS', 'WARN', 'FAIL'] },
    checks: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          name: { type: 'string' },
          passed: { type: 'boolean' },
          expected: { type: 'string' },
          actual: { type: 'string' },
        },
        required: ['name', 'passed', 'expected', 'actual'],
      },
    },
    issues: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['dimension', 'verdict', 'checks', 'issues', 'summary'],
}

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    dimension: { type: 'string' },
    agrees: { type: 'boolean' },
    recomputed_verdict: { type: 'string', enum: ['PASS', 'WARN', 'FAIL'] },
    disagreements: { type: 'array', items: { type: 'string' } },
    missed_issues: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    summary: { type: 'string' },
  },
  required: ['dimension', 'agrees', 'recomputed_verdict', 'disagreements', 'missed_issues', 'confidence', 'summary'],
}

const DIMENSIONS = [
  {
    key: 'netting',
    task: `POSITION NETTING & AGGREGATION CORRECTNESS.
Load parse_goldman.csv, parse_ibkr.csv, parse_merged.csv. Rebuild the expected merge yourself:
group both source frames by merge key (EQ:<underlying> for equities, OPT:<contract_key> for options),
sum qty, drop net-zero keys. Then verify:
- every merged key's qty EXACTLY equals your recomputed sum;
- no merged key is missing from / extra vs the union of source keys (minus net-zero drops);
- merged position count == number of non-zero summed keys;
- specifically confirm at least 3 keys that appear in BOTH files were summed (show the arithmetic),
  and confirm any net-zero cancellation was dropped (if none exist, state that).`,
  },
  {
    key: 'cash_nav_aum',
    task: `CASH / NAV / AUM HANDLING.
From the meta jsons and summary.json verify:
- merged cash == IBKR file cash (Goldman reports none);
- merged nav is None (guard: not all files report NAV);
- analytics summary aum == mv_net + cash (recompute mv_net = mv_long + mv_short from positions.csv);
- AUM sign/magnitude is sane. FLAG (WARN) that Goldman cash is absent, so the consolidated AUM
  omits Goldman's cash balance and is therefore understated — confirm this is the actual behavior
  and quantify roughly how large the distortion is vs gross exposure.`,
  },
  {
    key: 'asof',
    task: `AS-OF DATE HANDLING.
Verify merged asof == max(Goldman 2026-07-07, IBKR 2026-07-10) == 2026-07-10; that a "different
as-of dates" issue was appended to merged issues (check parse_merged_meta.json issues); and that the
pipeline used 2026-07-10 downstream (check pipeline_log.txt "as of" line and the PDF/report date).
Assess the risk: Goldman positions are stale by 1 trading day vs IBKR — is that disclosed anywhere?`,
  },
  {
    key: 'accounts_meta',
    task: `ACCOUNTS & SOURCE METADATA.
Verify merged accounts == sorted union of the two source accounts (expect AWKF1209 + U12004488),
source label == "goldman+ibkr", and the report name is "Consolidated GS+IBKR". Confirm both accounts'
positions actually survived into the merged book (count per account from parse_merged.csv).`,
  },
  {
    key: 'exposure_consistency',
    task: `EXPOSURE SUMMARY SELF-CONSISTENCY.
From summary.json and positions.csv recompute and check:
- exp_gross == exp_long - exp_short; exp_net == exp_long + exp_short;
- opt_exp_* + eq_exp_* splits reconcile to totals (eq_exp_net + opt_exp_net == exp_net within tol);
- mv_gross/mv_net consistent with per-position mv sums;
- beta_net and beta_coverage present and finite; n_instruments == rows in positions.csv;
- n_issuers == unique underlyings in issuers.csv.`,
  },
  {
    key: 'issuer_overlap',
    task: `ISSUER-LEVEL OVERLAP (cross-broker netting into one issuer).
Find underlyings that are EQUITY-held in BOTH source files. For 3-5 of them, verify the merged
issuer-level net share/qty and exposure reflect goldman_qty + ibkr_qty netted (a long in one broker
and short in the other must partially cancel at the issuer level). Reconcile issuers.csv against your
hand-summed quantities. Confirm options on a name net against stock in the same issuer row.`,
  },
  {
    key: 'factor_risk_sanity',
    task: `FACTOR MODEL & RISK SANITY.
From factor_risk.json and scenarios.json check: vol_total > 0 and finite; vol_factor & vol_specific
combine sensibly (factor_var_share in [0,1]); model coverage high (>0.9); every net factor exposure
finite; VaR_95 <= VaR_99 <= gross exposure and all positive; ES_95 >= VaR_95. Sanity-scale predicted
vol and VaR against gross/net exposure (are they plausible fractions, not absurd?).`,
  },
  {
    key: 'data_quality',
    task: `DATA-QUALITY & UNPRICED HANDLING.
Confirm the 4 Goldman blank-symbol parse issues are surfaced in issues.json; confirm the IBKR .CVR
contingent-value-right names (MRTX-CVR, APLS-CVR, ACLX-CVR, CNTA-CVR) are excluded/flagged rather than
silently mispriced; confirm priced coverage from pipeline_log.txt (388/392 tickers) and the option
chain match rate are reasonable. Ensure nothing unpriced silently entered exposures (positions.csv
rows with null price should not contribute to mv/exposure).`,
  },
  {
    key: 'pdf_integrity',
    task: `PDF / OUTPUT INTEGRITY.
From pdf_info.json confirm the PDF exists, size is non-trivial (> 20 KB), and page count is 2 (report
+ factor page). If pypdf is available, open the PDF and confirm it parses. Confirm the filename/date
match the 2026-07-10 as-of and the "Consolidated GS+IBKR" name. Note any render error in the log.`,
  },
]

phase('Verify')
const results = await pipeline(
  DIMENSIONS,
  (d) => agent(`${COMMON}\n\n### YOUR DIMENSION: ${d.key}\n${d.task}`, {
    label: `verify:${d.key}`, phase: 'Verify', schema: VERIFY_SCHEMA,
  }),
  (verify, d) => {
    if (!verify) return null
    return agent(
      `${COMMON}\n\n### AUDIT of dimension "${d.key}".\nAnother agent already verified this dimension and returned:\n` +
      JSON.stringify(verify, null, 1) +
      `\n\nDo NOT trust it. Independently RECOMPUTE the key numbers for this dimension from the raw files/artifacts ` +
      `and decide whether you AGREE. Try to REFUTE its verdict — look for an error in its arithmetic, a check it ` +
      `skipped, or a defect it rationalized as "expected". Report your own recomputed verdict and any missed issues.\n\n` +
      `Dimension task for reference:\n${d.task}`,
      { label: `audit:${d.key}`, phase: 'Audit', schema: AUDIT_SCHEMA },
    ).then((audit) => ({ dimension: d.key, verify, audit }))
  },
)

return { results: results.filter(Boolean) }
