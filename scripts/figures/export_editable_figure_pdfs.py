#!/usr/bin/env python3
"""Crop a LibreOffice-rendered native PPTX PDF and verify figure editability.

The cropped PDFs reuse vector page content. This script never rasterizes paper
figures; PNG output is exclusively for visual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import fitz
import pikepdf

NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest_path = directory / 'figure_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    pptx = directory / 'CBMJev_Figures.pptx'
    combined = directory / 'CBMJev_Figures.pdf'
    report = {'pptx_sha256': sha(pptx), 'combined_pdf_sha256': sha(combined), 'figures': []}
    preview = directory / 'previews'
    preview.mkdir(exist_ok=True)
    with zipfile.ZipFile(pptx) as package, fitz.open(combined) as source:
        if len(source) != len(manifest['figures']):
            raise ValueError('PPTX and PDF slide counts differ')
        for fig in manifest['figures']:
            index = fig['slide']
            root = ET.fromstring(package.read(f'ppt/slides/slide{index}.xml'))
            pictures = root.findall('.//p:pic', NS)
            if pictures:
                raise ValueError(f'Slide {index} contains picture objects instead of native marks')
            sizes, fonts, runs = set(), set(), 0
            for run in root.findall('.//a:r', NS):
                txt = run.find('a:t', NS)
                if txt is None or not (txt.text or '').strip():
                    continue
                prop = run.find('a:rPr', NS)
                if prop is None or 'sz' not in prop.attrib:
                    raise ValueError(f'Slide {index}: text has no explicit font size')
                sizes.add(int(prop.attrib['sz']) / 100)
                latin = prop.find('a:latin', NS)
                if latin is None:
                    raise ValueError(f'Slide {index}: text has no explicit font family')
                fonts.add(latin.attrib['typeface'])
                runs += 1
            if sizes != set(fig['font_sizes_pt']) or fonts != {'Arial'}:
                raise ValueError(f'Slide {index}: unexpected font policy {sizes}, {fonts}')
            slide = source[index - 1]
            crop = fig['crop_px']
            rect = fitz.Rect(crop['left'] * .75, crop['top'] * .75,
                             (crop['left'] + crop['width']) * .75,
                             (crop['top'] + crop['height']) * .75)
            dest = directory / (fig['id'] + '.pdf')
            with fitz.open() as doc:
                page = doc.new_page(width=rect.width, height=rect.height)
                page.show_pdf_page(page.rect, source, index - 1, clip=rect)
                doc.set_metadata({'title': fig['id'], 'subject': manifest['empirical_scope'],
                                  'creator': 'CBMJev native editable PPTX figure pipeline'})
                with pikepdf.Pdf.open(io.BytesIO(doc.tobytes(garbage=4, deflate=True))) as compatible:
                    compatible.save(dest, force_version='1.5')
            with fitz.open(dest) as doc:
                page = doc[0]
                if page.get_images(full=True):
                    raise ValueError(f'{dest.name}: raster image found in vector PDF')
                page.get_pixmap(matrix=fitz.Matrix(3, 3)).save(preview / (fig['id'] + '.png'))
                text = page.get_text()
                if len(text.strip()) < 30:
                    raise ValueError(f'{dest.name}: text extraction failed')
                spans = [span for b in page.get_text('dict')['blocks'] if b['type'] == 0
                         for line in b['lines'] for span in line['spans']]
                clipped = [span['text'] for span in spans
                           if not (page.rect + (-.5, -.5, .5, .5)).contains(fitz.Rect(span['bbox']))]
                if clipped:
                    raise ValueError(f'{dest.name}: clipped text {clipped}')
                font_names = sorted({span['font'] for span in spans})
                if any(not name.startswith('Arial') for name in font_names):
                    raise ValueError(f'{dest.name}: Arial was substituted: {font_names}')
                fonts_embedded = all(doc.extract_font(font[0])[3] for font in page.get_fonts(full=True))
                if not fonts_embedded:
                    raise ValueError(f'{dest.name}: an unembedded font was found')
                item = {'id': fig['id'], 'slide': index, 'native_shapes': len(root.findall('.//p:sp', NS)),
                        'text_runs': runs, 'font_sizes_pt': sorted(sizes), 'fonts': sorted(fonts),
                        'pdf_fonts': font_names, 'all_fonts_embedded': fonts_embedded,
                        'raster_images': 0, 'pdf_sha256': sha(dest), 'text': text,
                        'width_pt': page.rect.width, 'height_pt': page.rect.height}
                report['figures'].append(item)
    report['status'] = 'PASS'
    (directory / 'figure_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'slides': len(report['figures']), 'directory': str(directory)}))


if __name__ == '__main__':
    main()
