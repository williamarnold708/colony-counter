# Colony Labeller

Builds a YOLO training dataset from your plate photos.

## Use

1. Drop plate photos (JPG/PNG) into `inbox/`.
2. `python app.py`, then open <http://127.0.0.1:5001>.
3. For each plate: click each colony to mark it, click a mark to remove it.
   Auto-detection pre-fills what it can, so you mostly confirm.
   - <kbd>S</kbd> save & next, <kbd>K</kbd> skip, <kbd>Z</kbd> undo, <kbd>C</kbd> clear
4. Progress saves as you go; reopening resumes where you left off.

## Output

```
data/images/<name>.jpg     data/labels/<name>.txt   (YOLO format)
data/classes.txt           data/progress.json
```

## Tips for a good dataset

- Label consistently — decide your rule for borderline specks and keep to it.
- Skip confluent lawns (<kbd>K</kbd>); they aren't countable.
- Include hard cases (gridded, faint) — they're the most valuable examples.
- Variety over volume: different media, colony types, lighting.

## Classes

Single class (`colony`) by default. To label colony *types*, edit
`data/classes.txt` (one name per line); the UI shows class buttons
(<kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd>). Point positions are unaffected, so
you can start single-class and add types later.
