# Running the breakout forward test in the cloud (GitHub Actions)

The workflow in `.github/workflows/paper-trade.yml` advances the breakout paper
account once per trading day, commits the updated log back to the repo, and
publishes the dashboards to GitHub Pages — no server to run.

## What it does each run
1. Checks out the repo (including the latest `data/paper_breakout.json`).
2. Runs `breakout_dashboard.py` → advances the paper account, logs any trades
   (with the momentum + news sentiment captured at entry) and daily equity.
3. Commits `data/paper_breakout.json`, `paper_trades.csv`, `paper_equity.csv`
   back to the repo (so the forward record persists and is auditable in git).
4. Publishes `index.html` + both dashboards to GitHub Pages.

## One-time setup
1. **Create a GitHub repo and push this project** (the schedule only fires on the
   repo's **default branch**, so push this to `main`):
   ```bash
   gh repo create Stock_trading_agent --private --source=. --push   # or add a remote manually
   ```
2. **Settings → Actions → General → Workflow permissions:** select
   *Read and write permissions* (lets the job commit the log back).
3. **Settings → Pages → Build and deployment → Source:** *GitHub Actions*.
4. **Test it now:** Actions tab → *Daily paper trade* → **Run workflow**. Check the
   Pages URL it prints, and that a "Paper trade update …" commit appears.

## Schedule / timezone
- Cron is `30 22 * * 1-5` = **22:30 UTC, weekdays** (~6:30pm ET, after the US close
  so the day's data exists). GitHub cron is **UTC only** and can be delayed a few
  minutes under load. Adjust the line if you want a different time.

## Caveats
- **yfinance from cloud IPs** is occasionally rate-limited/blocked. `load_prices`
  retries, but a run can still fail — it simply resumes the next day (no data is
  lost; the paper state only advances on successful runs). If it proves flaky, a
  small always-on VPS with a `cron` job is more reliable.
- **No secrets needed** today (sentiment uses local VADER). To upgrade to
  Claude-API sentiment later, add `ANTHROPIC_API_KEY` as a repo secret and read it
  in `sentiment_score()`.
- The account **start date and $8,800 capital are frozen** in the committed
  `data/paper_breakout.json`; delete that file to restart the forward test.
