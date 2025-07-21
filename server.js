const express = require('express');
const fetch = require('node-fetch');
const sharp = require('sharp');
const { createWorker } = require('tesseract.js');
const cors = require('cors');

const app = express();

let tesseractWorker;
let isTesseractWorkerReady = false;

async function initializeTesseractWorker() {
    console.log("Initializing Tesseract.js worker...");
    try {
        const worker = await createWorker();
        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        await worker.setParameters({
            tessedit_char_whitelist: '0123456789',
            tessedit_pageseg_mode: '8', // single word/digit line mode (faster, accurate for captchas)
            user_defined_dpi: '300',
        });
        tesseractWorker = worker;
        isTesseractWorkerReady = true;
        console.log("Tesseract.js initialized.");
    } catch (err) {
        console.error("Failed to initialize Tesseract.js:", err);
        process.exit(1);
    }
}

const tesseractReadyPromise = initializeTesseractWorker();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.post('/solve-captcha', async (req, res) => {
    await tesseractReadyPromise;
    if (!isTesseractWorkerReady) {
        return res.status(503).json({ error: 'OCR worker initializing.' });
    }

    const { imageUrl } = req.body;
    if (!imageUrl) {
        return res.status(400).json({ error: 'Missing imageUrl in request.' });
    }

    try {
        let imageBuffer;

        // Handle Base64 or remote image
        if (imageUrl.startsWith('data:image')) {
            const base64 = imageUrl.split(',')[1];
            imageBuffer = Buffer.from(base64, 'base64');
        } else if (imageUrl.startsWith('http')) {
            const response = await fetch(imageUrl);
            if (!response.ok) throw new Error(`Image fetch failed: ${response.statusText}`);
            imageBuffer = await response.buffer();
        } else {
            throw new Error('Invalid image URL format.');
        }

        // Preprocess: resize, greyscale, sharpen
        const preprocessed = await sharp(imageBuffer)
            .resize({ width: 180 })
            .greyscale()
            .sharpen()
            .median(3)
            .raw()
            .toBuffer({ resolveWithObject: true });

        const { data: rawData, info } = preprocessed;
        const { width, height } = info;

        // Detect brightness of center region (ROI)
        const centerBox = {
            x1: Math.floor(width * 0.25),
            x2: Math.floor(width * 0.75),
            y1: Math.floor(height * 0.25),
            y2: Math.floor(height * 0.75)
        };

        let sum = 0;
        let count = 0;
        for (let y = centerBox.y1; y < centerBox.y2; y++) {
            for (let x = centerBox.x1; x < centerBox.x2; x++) {
                sum += rawData[y * width + x];
                count++;
            }
        }
        const avgGray = count ? sum / count : 128;
        const numbersAreLighter = avgGray > 128;

        // Reconstruct sharp from raw, apply threshold (invert if needed)
        let sharpImage = sharp(rawData, {
            raw: { width, height, channels: 1 }
        });

        if (numbersAreLighter) {
            sharpImage = sharpImage.negate();
        }

        const threshold = otsuThreshold(rawData);
        const finalImageBuffer = await sharpImage
            .threshold(threshold)
            .toFormat('png')
            .toBuffer();

        // OCR with Tesseract
        const result = await tesseractWorker.recognize(finalImageBuffer);
        const cleanedText = result.data.text.replace(/[^\d]/g, '');

        console.log(`Detected: "${result.data.text.trim()}", Cleaned: "${cleanedText}"`);
        res.json({ text: cleanedText });

    } catch (err) {
        console.error("CAPTCHA solve error:", err);
        res.status(500).json({ error: 'OCR processing failed.' });
    }
});

function otsuThreshold(data) {
    const hist = new Uint32Array(256);
    let total = data.length, sum = 0;

    for (let i = 0; i < total; i++) {
        const val = data[i];
        hist[val]++;
        sum += val;
    }

    let sumB = 0, wB = 0, wF = 0, varMax = 0, threshold = 0;
    for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;

        wF = total - wB;
        if (wF === 0) break;

        sumB += t * hist[t];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const between = wB * wF * (mB - mF) ** 2;

        if (between > varMax) {
            varMax = between;
            threshold = t;
        }
    }
    return threshold;
}

module.exports = app;
