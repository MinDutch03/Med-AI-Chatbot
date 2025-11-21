import os
import json
from glob import glob
import xml.etree.ElementTree as ET

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEDQUAD_DIR = os.path.join(BASE_DIR, 'MedQuAD-master')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'medquad', 'processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_medquad.json')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all numbered folders (e.g., 1_CancerGov_QA, 2_GARD_QA, ...)
folders = [f for f in os.listdir(MEDQUAD_DIR) if os.path.isdir(os.path.join(MEDQUAD_DIR, f)) and f[0].isdigit()]

all_qa = []

for folder in folders:
    folder_path = os.path.join(MEDQUAD_DIR, folder)
    xml_files = glob(os.path.join(folder_path, '*.xml'))
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            doc_source = root.attrib.get('source', folder)
            doc_url = root.attrib.get('url', '')
            focus_elem = root.find('Focus')
            focus = focus_elem.text.strip() if focus_elem is not None else ''
            qapairs = root.find('QAPairs')
            if qapairs is not None:
                for qapair in qapairs.findall('QAPair'):
                    question_elem = qapair.find('Question')
                    answer_elem = qapair.find('Answer')
                    question = question_elem.text.strip() if question_elem is not None and question_elem.text else ''
                    answer = answer_elem.text.strip() if answer_elem is not None and answer_elem.text else ''
                    if question and answer:
                        qa = {
                            'question': question,
                            'answer': answer,
                            'source': doc_source,
                            'url': doc_url,
                            'focus': focus,
                            'file': os.path.basename(xf)
                        }
                        all_qa.append(qa)
        except Exception as e:
            print(f"Error reading {xf}: {e}")

# Save combined data
with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
    json.dump(all_qa, out_f, ensure_ascii=False, indent=2)

print(f"Processed {len(all_qa)} Q&A pairs. Saved to {OUTPUT_FILE}") 