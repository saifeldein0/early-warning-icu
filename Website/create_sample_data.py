import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_DIR = os.path.join(BASE_DIR, "demo_data")


def create_sample_excel():
    """Create synthetic export files for the public website demo."""
    np.random.seed(57)

    first_names = [
        "Avery", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Parker", "Quinn", "Rowan", "Emerson",
        "Harper", "Finley", "Dakota", "Skyler", "Reese", "Cameron", "Hayden", "Sawyer", "Kendall", "Logan"
    ]
    last_names = [
        "Stone", "Hayes", "Brooks", "Reed", "Shaw", "Perry", "Lane", "Wells", "Cross", "Blake",
        "Cole", "Parker", "Flynn", "Sutton", "Bennett", "Foster", "Quincy", "Sloane", "Ellis", "Rowe"
    ]

    patient_names = [f"{first_names[i]} {last_names[i]}" for i in range(20)]
    sample_data = {
        "PATIENT_NAME": patient_names,
        "MRN": [f"DEMO-{i + 1:04d}" for i in range(20)],
        "LOCATION": [
            "Ward A", "Ward B", "Ward C", "Ward D", "Ward A",
            "Ward B", "Ward C", "Ward D", "Ward A", "Ward B",
            "Ward C", "Ward D", "Ward A", "Ward B", "Ward C",
            "Ward D", "Ward A", "Ward B", "Ward C", "Ward D"
        ],
        "ROOM": [
            "A-301", "B-402", "C-205", "D-503", "A-302",
            "B-401", "C-203", "D-504", "A-303", "B-404",
            "C-201", "D-505", "A-304", "B-403", "C-202",
            "D-501", "A-305", "B-406", "C-204", "D-502"
        ],
        "ADMISSION_DATE": pd.date_range("2026-02-01", periods=20, freq="12h").strftime("%Y-%m-%d %H:%M"),
        "ADMISSION_ORDER": [
            "Admission Hematology", "Admission Oncology", "Admission Hematology", "Admission Surgery",
            "Admission Hematology", "Admission Oncology", "Admission Hematology", "Admission Surgery",
            "Admission Hematology", "Admission Oncology", "Admission Hematology", "Admission Surgery",
            "Admission Hematology", "Admission Oncology", "Admission Hematology", "Admission Surgery",
            "Admission Hematology", "Admission Oncology", "Admission Hematology", "Admission Surgery"
        ],
        "DIAGNOSIS": [
            "Synthetic leukemia case", "Synthetic oncology observation", "Synthetic anemia review", "Synthetic neurosurgery recovery",
            "Synthetic lymphoma review", "Synthetic oncology follow-up", "Synthetic neutropenia monitoring", "Synthetic surgical observation",
            "Synthetic marrow suppression", "Synthetic oncology escalation", "Synthetic transfusion monitoring", "Synthetic post-op recovery",
            "Synthetic relapse surveillance", "Synthetic oncology stabilization", "Synthetic infection workup", "Synthetic respiratory support",
            "Synthetic hematology surveillance", "Synthetic oncology discharge planning", "Synthetic platelet monitoring", "Synthetic hemodynamic review"
        ]
    }

    vital_signs = [
        "HEART_RATE", "PULSE_OXIMETRY", "TEMPERATURE",
        "SYSTOLIC_BLOOD_PRESSURE", "MEAN_ARTERIAL_PRESSURE",
        "DIASTOLIC_BLOOD_PRESSURE", "RESPIRATION_RATE"
    ]

    for col in vital_signs:
        sample_data[col] = []
        for _ in range(20):
            if col == "HEART_RATE":
                values = [f"{np.random.randint(60, 130)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "PULSE_OXIMETRY":
                values = [f"{np.random.randint(90, 100)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]
            elif col == "TEMPERATURE":
                values = [f"{np.random.randint(36, 39)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]
            elif col == "SYSTOLIC_BLOOD_PRESSURE":
                values = [f"{np.random.randint(95, 160)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]
            elif col == "MEAN_ARTERIAL_PRESSURE":
                values = [f"{np.random.randint(60, 110)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]
            elif col == "DIASTOLIC_BLOOD_PRESSURE":
                values = [f"{np.random.randint(55, 95)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]
            else:
                values = [f"{np.random.randint(12, 28)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 4))]

            sample_data[col].append(",".join(values))

    lab_cols = [
        "AST_RESULT", "CREATININE_RESULT", "TOTAL_BILIRUBIN_RESULT",
        "DIRECT_BILIRUBIN_RESULT", "POTASSIUM_RESULT", "HEMOGLOBIN_RESULT",
        "LEUKOCYTE_COUNT_RESULT", "ABSOLUTE_NEUTROPHILS", "PLATELET_COUNT_RESULT",
        "PROTHROMBIN_CONCENTRATION"
    ]

    for col in lab_cols:
        sample_data[col] = []
        for _ in range(20):
            if col == "AST_RESULT":
                values = [f"{np.random.randint(10, 55)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "CREATININE_RESULT":
                values = [f"{np.random.randint(6, 20)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "TOTAL_BILIRUBIN_RESULT":
                values = [f"{np.random.randint(1, 8)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "DIRECT_BILIRUBIN_RESULT":
                values = [f"{np.random.randint(0, 4)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "POTASSIUM_RESULT":
                values = [f"{np.random.randint(35, 50)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "HEMOGLOBIN_RESULT":
                values = [f"{np.random.randint(8, 16)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "LEUKOCYTE_COUNT_RESULT":
                values = [f"{np.random.randint(3000, 16000)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "ABSOLUTE_NEUTROPHILS":
                values = [f"{np.random.randint(1500, 9000)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            elif col == "PLATELET_COUNT_RESULT":
                values = [f"{np.random.randint(70000, 450000)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]
            else:
                values = [f"{np.random.randint(65, 120)}.{np.random.randint(0, 9)}" for _ in range(np.random.randint(2, 5))]

            sample_data[col].append(",".join(values))

    med_cols = ["ANTIBIOTICS", "NEUROLOGY_DRUGS", "CARDIOLOGY_DRUGS", "FUNGAL_DRUGS"]
    antibiotics = ["amoxicillin", "ceftriaxone", "vancomycin", "piperacillin", "meropenem"]
    neurology_drugs = ["levetiracetam", "phenytoin", "valproate", "lorazepam", "midazolam"]
    cardiology_drugs = ["metoprolol", "amlodipine", "lisinopril", "furosemide", "digoxin"]
    fungal_drugs = ["fluconazole", "amphotericin", "caspofungin", "voriconazole", "micafungin"]
    med_lists = [antibiotics, neurology_drugs, cardiology_drugs, fungal_drugs]

    for idx, col in enumerate(med_cols):
        sample_data[col] = []
        for _ in range(20):
            num_meds = np.random.randint(0, 4)
            if num_meds > 0:
                selected_meds = np.random.choice(med_lists[idx], size=num_meds, replace=False)
                sample_data[col].append(",".join(selected_meds))
            else:
                sample_data[col].append("")

    df = pd.DataFrame(sample_data)

    os.makedirs(DEMO_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DEMO_DATA_DIR, "export_all_columns_demo.csv")
    xlsx_path = os.path.join(DEMO_DATA_DIR, "export_all_columns_demo.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print("Synthetic public demo files created:")
    print(f"- {csv_path}")
    print(f"- {xlsx_path}")
    print(f"Each file contains {len(df)} synthetic patients with the website input schema.")


if __name__ == "__main__":
    create_sample_excel()
