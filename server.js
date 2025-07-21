const express = require('express');
const fetch = require('node-fetch');
const sharp = require('sharp');
const { createWorker } = require('tesseract.js');
const cors = require('cors');
const path = require('path');

const app = express();

let tesseractWorker;
let isTesseractWorkerReady = false;

async function initializeTesseractWorker() {
    console.log("Initializing Tesseract.js worker with local model...");
    try {
        const worker = await createWorker();
        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        await worker.setParameters({
            tessedit_char_whitelist: '0123456789',
            tessedit_pageseg_mode: 7,
            oem: 3,
        });
        tesseractWorker = worker;
        isTesseractWorkerReady = true;
        console.log("SUCCESS: Tesseract.js worker initialized.");
    } catch (error) {
        console.error("ERROR: Failed to initialize Tesseract.js worker.", error);
        process.exit(1);
    }
}

const tesseractReadyPromise = initializeTesseractWorker();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.post('/solve-captcha', async (req, res) => {
    await tesseractReadyPromise;
    if (!isTesseractWorkerReady) {
        return res.status(503).json({ error: 'Server is still initializing. Please try again shortly.' });
    }

    const { imageUrl } = req.body;
    if (!imageUrl) {
        return res.status(400).json({ error: 'imageUrl is required in the request body.' });
    }

    try {
        let imageBuffer;

        if (imageUrl.startsWith('data:image')) {
            const base64Data = imageUrl.split(',')[1];
            imageBuffer = Buffer.from(base64Data, 'base64');
        } else if (imageUrl.startsWith('http')) {
            const imageResponse = await fetch(imageUrl);
            if (!imageResponse.ok) throw new Error(`Failed to fetch image. Status: ${imageResponse.statusText}`);
            imageBuffer = await imageResponse.buffer();
        } else {
            throw new Error('Invalid imageUrl format.');
        }

        let sharpInstance = sharp(imageBuffer)
            .resize({ width: 180 })
            .greyscale()
            .sharpen()
            .median(4);

        const { data, info } = await sharpInstance.raw().toBuffer({ resolveWithObject: true });
        const { width, height } = info;

        const roiStartX = Math.floor(width * 0.25);
        const roiEndX = Math.floor(width * 0.75);
        const roiStartY = Math.floor(height * 0.25);
        const roiEndY = Math.floor(height * 0.75);

        let roiSum = 0;
        let roiCount = 0;
        for (let y = roiStartY; y < roiEndY; y++) {
            for (let x = roiStartX; x < roiEndX; x++) {
                roiSum += data[y * width + x];
                roiCount++;
            }
        }

        const avgROIGray = roiCount > 0 ? roiSum / roiCount : 128;
        const threshold = 128;
        const numbersAreLighter = avgROIGray > threshold;

        sharpInstance = sharp(data, {
            raw: {
                width,
                height,
                channels: 1,
            }
        });

        let finalImageBuffer;

        if (numbersAreLighter) {
            finalImageBuffer = await sharpInstance
                .negate()
                .threshold(otsuLike(data))
                .toFormat('png')
                .toBuffer();
        } else {
            finalImageBuffer = await sharpInstance
                .threshold(otsuLike(data))
                .toFormat('png')
                .toBuffer();
        }

        const { data: { text } } = await tesseractWorker.recognize(finalImageBuffer);
        const cleanedText = text.replace(/[\s\D]/g, '');

        console.log(`Raw Tesseract text: "${text.trim()}", Cleaned text: "${cleanedText}"`);
        res.json({ text: cleanedText });

    } catch (error) {
        console.error("Error during CAPTCHA processing:", error);
        res.status(500).json({ error: 'Failed to process the CAPTCHA image on the server.' });
    }
});

function otsuLike(grayData) {
    const hist = new Array(256).fill(0);
    for (let i = 0; i < grayData.length; i++) {
        hist[grayData[i]]++;
    }
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0, wB = 0, wF = 0, varMax = 0, threshold = 0;
    const total = grayData.length;
    for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;
        wF = total - wB;
        if (wF === 0) break;
        sumB += t * hist[t];
        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const betweenVar = wB * wF * (mB - mF) ** 2;
        if (betweenVar > varMax) {
            varMax = betweenVar;
            threshold = t;
        }
    }
    return threshold;
}

module.exports = app;
