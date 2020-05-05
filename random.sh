cd OCR-D_IMG_OLD
find . -type f | \
	shuf -n 208 | \
	while read -r fn; do
		mv "$fn" train
		s=${fn##*/}
		mv "../OCR-D_LABELS_OLD/$s" "../OCR-D_LABELS_OLD/train"
		mv "../OCR-D_GT_OLD/${s%.png}.xml" "../OCR-D_GT_OLD/train"
	done
