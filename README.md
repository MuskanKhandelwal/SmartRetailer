# SmartRetailer

> Multi-agent dynamic pricing on public retail data with a simulator seeded by real transactions.

## Project Summary

Small retailers often rely on static markups and ad-hoc promos, missing revenue when demand shifts with seasonality, inflation, or competitor moves. This project prototypes **learned pricing agents** that experiment safely in a **market simulator** and choose prices that increase revenue while respecting **fairness caps**. We benchmark against **static** and **rule-based** baselines.

This repo is tailored to **Dominick’s weekly store-level** scanner data. We build a clean soft-drinks panel from the classic files (upcsdr.csv, wsdr.csv, dominicks_weeks.csv), then will run baselines and agents. Agents try small price changes in a sandbox and keep those that help revenue without violating caps.
