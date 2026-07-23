# Local image data

Place image datasets anywhere under this directory. Nested folders are
discovered recursively by the development server.

For example:

```text
data/
├── train/
│   ├── image-001.png
│   └── image-002.png
├── val/
│   └── image-001.png
└── test/
    └── image-001.png
```

The browser only lists images directly inside the selected folder. Use the
folder selector or breadcrumbs to move through deeper structures.
