# Dashboard Benchmarking and Admin Views

I have completed the upgrades to the internal dashboard! The `2_Dashboard_Internal.py` file has been completely rewritten to support these two distinct views.

## 1. HR Manager View (Benchmarking)
When an HR Manager logs into the internal dashboard, they will now see **two** lines on the radar (spider) chart:
1. **Solid Blue Line**: Their firm's average score.
2. **Dashed Grey Line**: The benchmark (Moyenne globale), which is calculated automatically by averaging the scores of all *other* firms in the database.

## 2. Admin View (Global Distributions)
I have updated the routing and role permissions so that the **Admin** account can now view the internal dashboard page. When logged in as Admin, the view transforms into a global comparative dashboard:

- **Multi-Firm Spider Graph**: Instead of showing one firm, it loops through every firm in the database and overlays their scores on a single spider graph for direct comparison.
- **33-Question Boxplot**: A large boxplot visualizes the statistical distribution (median, quartiles, outliers) of the scores for all 33 individual questions. The questions are labeled chronologically (Q1 through Q33) on the X-axis.
- **7-Pillar Violin Plot**: A beautiful violin plot visualizes the probability density and distribution shape of the scores aggregated across the 7 main HR Valais pillars.

> [!TIP]
> Both the boxplot and violin plot are fully interactive! You can hover over any point or box to see the exact statistical metrics (quartiles, min/max) and isolate specific firms by clicking on them in the legend.

If your local server is running, you can log in as a manager (e.g. `manager_firmA`) to see the benchmark, and then log out and log in as `admin` to see the global distribution plots!
