# Business Insights Report

## Objective
To understand campaign performance and provide actionable recommendations for future marketing strategies.

## Key Findings
- **High ROI Channels**: Certain channels drive higher engagement and conversions leading to a better profit margin. Identifying and doubling down on these channels is recommended.
- **Cost vs Revenue**: Some campaigns have a high `Acquisition_Cost` but fail to yield proportional `Revenue`. Predictive modeling will help identify such campaigns before heavy investment.
- **Brand Strategy**: The performance metrics are distributed differently among Nykaa, Purplle, and Tira. Tailoring campaign types to specific brands might yield better engagement.

## Recommendations
1. Allocate more budget to campaigns predicted to be profitable (Profit_Flag = 1).
2. Use the regression model to forecast expected Revenue based on planned impressions, clicks, and channels.
3. Continually monitor campaigns that predict high acquisition costs with low revenue.
