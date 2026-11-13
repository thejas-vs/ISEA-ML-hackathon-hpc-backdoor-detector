import sys
from pathlib import Path
from pdf2image import convert_from_path

pdf_path = Path('hpc-backdoor-detector/reports/report.pdf')
out_dir = Path('hpc-backdoor-detector/reports/report_pages')
out_dir.mkdir(parents=True, exist_ok=True)
pages = convert_from_path(str(pdf_path), fmt='png', output_folder=str(out_dir), dpi=150)
print('Converted', len(pages), 'pages')
