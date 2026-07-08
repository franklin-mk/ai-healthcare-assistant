# 🏥 Intelligent Healthcare Diagnostic Assistant AI

An integrated, multi-paradigm clinical diagnosis decision engine built for the **CCS 3101 Introduction to AI Capstone Project**. The system pairs classical expert rules with machine learning classification architectures to handle clinical reasoning, statistical patterns, uncertainty, and linear strategic treatment routes.

---

## 🛠️ System Architecture Diagram Overview

The core application pipes patient symptoms and vitals through a linear multi-module evaluation workflow:

```text
[Patient Percept Object Intake]
            │
            ├──► [Module 2] FOL Knowledge Base ──► Chaining Explanations
            ├──► [Module 3] Bayesian Network   ──► Joint Log Probabilities
            ├──► [Module 4] ML Tree Ensembles  ──► Feature Splits (Random Forest)
            ├──► [Module 5] Deep Neural Net    ──► Softmax Categorization
            │
            ▼
    [Module 6] Fuzzy Severity Assessor ──► Fuzzified Centroid Severity Score (0-100)
            │
            ▼
    [Module 7] STRIPS Search Planner   ──► turn-by-turn Linear Treatment Blueprint