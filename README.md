# Samotator hover demo

Local point-prompt segmentation using the supplied SAM3 Q4 ONNX bundle.

## Run

```bash
npm install
npm run dev
```

Open <http://127.0.0.1:5173> in a current Chrome or Edge browser with WebGPU
and hardware acceleration enabled.

The first run downloads the model files from the local Vite server into the
browser runtime. No remote model fallback is enabled. The 369 MB vision
encoder is loaded once, and the selected image is encoded once; pointer
movement only runs the smaller prompt decoder.

## Controls

- Move over the image for a temporary point preview.
- Choose **Positive** or **Negative** to change the temporary point type.
- Click to pin the current point.
- Use **Undo** or **Clear all** to revise pinned prompts.

The files under `sam3-q4/` must remain in their existing layout because both
ONNX graphs reference their adjacent `.onnx_data` files by name.
