# test_fuzzy.py
from modules.fuzzy_controller import FuzzySeverityAssessor

def run_fuzzy_verification():
    print("🚀 Initializing CCS 3101 Fuzzy Severity Assessor Unit Test...")
    
    # 1. Instantiate the fuzzy controller
    fuzz = FuzzySeverityAssessor()
    
    # 2. Mock a patient experiencing acute borderline vitals
    test_temp = 38.8  # Mildly high fever
    test_hr = 108    # Elevated heart rate (Tachycardia zone)
    test_symptoms = 6  # Large cluster of symptom nodes
    
    print(f"\n🧪 Testing Fuzzification & Inference Pipeline with:")
    print(f"  • Vitals Intake: Temp={test_temp}°C, HR={test_hr} BPM, Symptoms Count={test_symptoms}")
    
    # 3. Process the assessment metrics
    output = fuzz.assess(test_temp, test_hr, test_symptoms)
    
    print("\n📊 Generated Linguistic Inference Payload:")
    print("─" * 60)
    print(f"🎯 Assigned Severity Label : {output['severity_label']}")
    print(f"🎛️ Computed Defuzzed Score : {output['severity_score']} / 100")
    print("─" * 60)
    
    print("\n🔥 Evaluated Rule Strengths Matrix (Active Bounds):")
    for rule, strength in output['rule_strengths'].items():
        print(f"  • Rule [{rule:<8}] fired with truth value: {strength:.3f}")

if __name__ == "__main__":
    run_fuzzy_verification()