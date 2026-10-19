# Comparing OCR methods

The same YOLO plate detector runs once per comparison. Each detected, clipped
plate crop is copied without modifications and supplied to every selected OCR
adapter. Detection confidence and crop SHA-256 identify the common input.

| Dropdown method | Recognizer / configuration | Runtime |
| --- | --- | --- |
| Plate OCR · CCT | CCT-XS-v2 global plate recognizer; configured custom character weights replace this slot when present | ONNX Runtime CPU, or configured custom runtime |
| EasyOCR | English generation-2 recognition; `recognize` on the whole detected crop; greedy decoder and A–Z/0–9 allowlist | PyTorch CPU |
| Tesseract | English LSTM; page segmentation mode 6, A–Z/0–9 whitelist | Tesseract CPU executable |
| PaddleOCR | `en_PP-OCRv5_mobile_rec` through PaddleOCR `TextRecognition` | PaddlePaddle CPU |

EasyOCR's additional text detector is disabled because YOLO already supplies the
plate regions. CCT, EasyOCR, and PaddleOCR perform their own recognizer resizing
and normalization internally. Tesseract converts to grayscale, scales toward
80 pixels in height by at most 3×, applies Otsu thresholding with dark text on
white, and adds a 10-pixel white border. These are fixed method settings, not chosen
separately after seeing the expected number. Perspective rectification, automatic
two-line splitting, and character substitutions are not applied.

## Use the dashboard

1. Upload a photo or pause at a camera/video frame.
2. Select an OCR method and Analyze for normal recognition. New history entries
   include that method; switching methods resets the browser's recognition session.
3. For a comparison, optionally type the manually verified plate into Expected plate.
   Scoring one expected number requires exactly one detected plate. For several
   vehicles, crop to one plate or leave Expected plate empty.
4. Select Compare methods. It runs every installed method, reports errors for
   individual failed engines, and never substitutes another engine's reading.
5. Export comparison CSV. It contains raw and normalized text, method, per-plate
   OCR time, model setup time, common detection time, expected number, exact match,
   character error rate, crop SHA-256, and error details.

The interface keeps detection history on its existing separate page. Comparisons
are temporary and only saved when exported. Normal recognition stores selected
method results locally in SQLite; the database migration preserves older entries
and labels their original recognizer as CCT. Unconfirmed live observations from
different methods never share temporal consensus or duplicate suppression.

## Interpret results

- **OCR time:** one `perf_counter` measurement per method and plate, including
  method preprocessing, native execution, and decoding. Tesseract time includes
  launching its executable. Excludes shared plate detection, package/model loading,
  HTTP upload, drawing, and database writes.
- **Model setup:** initialization is displayed separately. The first prediction may
  still include runtime kernel/cache warmup. Run an unscored warmup first, repeat
  comparisons, and compare median timings on the same CPU.
- **Exact match:** normalized prediction equals the manually entered reference.
- **Character error rate (CER):** Levenshtein character edits divided by reference
  length. Lower is better. An empty reading has 100% CER; insertions can make CER
  exceed 100%. Both texts are uppercased and stripped to A–Z/0–9, without O/0 or
  I/1 guessing. Missing expected text leaves accuracy fields unscored.
- **Confidence:** method-specific model output, not a common accuracy scale. A high
  confidence reading can be incorrect. Do not rank OCR accuracy by confidence.
- **Failed engine:** an explicit error, not a guessed reading or a measured zero
  milliseconds. Record engine failures separately when reporting accuracy/coverage.

For a faculty report, use a fixed, manually transcribed set of held-out images
covering blur, lighting, angle, and plate layout. Keep input resolution and detector
threshold fixed; record model/package versions and hardware. Report exact plate
accuracy across all cases, aggregate CER (total edits / total reference characters),
read/failure coverage, and warmed median OCR time. Agreement among engines does not
establish correctness. The local single-image smoke check in
`data/ocr-verification.json` verifies integration and is not a dataset benchmark.

## Official references

- [EasyOCR API](https://jaided.ai/easyocr/documentation/)
- [Tesseract command-line usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html)
- [PaddleOCR text recognition models and API](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md)
