# app.py
# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — Unified Ecosystem Engine
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  

# Import all completed modules  
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules into the core agent container"""  
    banner()
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing diagnostic specialists...")  
    
    # Instantiate the clinical subsystems
    kb = MedicalKnowledgeBase()
    bn = SimpleBayesianDiagnostics()
    ml = MLDiagnosticClassifier()
    dnn = NeuralDiagnosticModel()
    fuzzy = FuzzySeverityAssessor()
    planner = TreatmentPlanner()

    # Pre-train machine learning components automatically on startup
    print("  Pre-training Supervised Classifiers...")
    ml.train(verbose=False)
    dnn.train(epochs=15, verbose=0)

    # Register each module to the agent using the template interface
    agent.register_module('KnowledgeBase', kb)  
    agent.register_module('BayesianNet',   bn)  
    agent.register_module('MLClassifier',  ml)  
    agent.register_module('NeuralNetwork', dnn)  
    agent.register_module('FuzzyLogic',    fuzzy)  
    agent.register_module('TreatmentPlan', planner)  

    print(f"\n✨ {C.GREEN}All components wired successfully into core engine!{C.END}")
    return agent  

def execute_clinical_triage():
    """Runs a complete test case evaluation against the integrated platform"""
    # 1. Build the integrated system
    agent = build_system()
    
    # 2. Mock a patient percept profile
    sample_patient = PatientPercept(
        patient_id="PT-2026",
        symptoms=["Fever", "Cough", "Loss of Smell", "Fatigue"],
        age=28,
        temperature=39.1,
        heart_rate=112,
        blood_pressure="125/85"
    )
    
    section("📥 Patient Intake Form Perceived")
    print(f"  Patient ID  : {sample_patient.patient_id}")
    print(f"  Symptoms    : {sample_patient.symptoms}")
    print(f"  Vitals Panel: {sample_patient.temperature}°C | {sample_patient.heart_rate} BPM")
    
    section("⚙️  Running Integrated AI Analysis Pipeline")
    
    # In order to let the structural STRIPS planner know what the consensus diagnosis is,
    # we inject helper placeholders dynamically before running the loop.
    sample_patient.diagnosis_guess = "covid19"
    sample_patient.urgency_guess = "HIGH"
    
    # Trigger the unified loop
    report = agent.run(sample_patient)
    
    section("📊 Final Consolidated Action Report Summary")
    print(f"  Consensus Diagnosis : {C.BOLD}{C.GREEN}{report['diagnosis'].upper()}{C.END}")
    print(f"  Aggregated Certainty: {report['confidence']:.2%}")
    print(f"  Triage Urgency Level: {C.BOLD}{C.RED}{report['urgency']}{C.END}")
    print(f"  Suggested Action    : {report['next_action']}")
    
    print(f"\n📋 {C.BOLD}Clinical Recommendation Protocols:{C.END}")
    for recommendation in report['recommendations']:
        print(f"   • {recommendation}")

    # Output tracking telemetry
    agent.print_log()

if __name__ == "__main__":  
    execute_clinical_triage()