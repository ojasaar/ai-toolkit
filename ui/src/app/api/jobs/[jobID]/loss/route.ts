import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { calculateMultipleEMAs } from '@/utils/ema';
import {
  downsampleLTTB,
  prepareDataForDownsampling,
  extractDownsampledData,
} from '@/utils/downsampling';

const prisma = new PrismaClient();

interface LossEntry {
  step: number;
  loss: number;
  learning_rate: number;
  timestamp: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { jobID: string } }
) {
  const { jobID } = await params;
  const { searchParams } = new URL(request.url);

  // Get optional query parameters
  const resolution = parseInt(searchParams.get('resolution') || '1500', 10);
  const startStep = parseInt(searchParams.get('start_step') || '0', 10);
  const endStep = parseInt(searchParams.get('end_step') || '-1', 10);

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const lossDataPath = path.join(jobFolder, 'loss_data.jsonl');

  if (!fs.existsSync(lossDataPath)) {
    return NextResponse.json({
      steps: [],
      losses: [],
      ema90: [],
      ema95: [],
      ema99: [],
    });
  }

  try {
    // Read and parse JSONL file
    const fileContent = fs.readFileSync(lossDataPath, 'utf-8');
    const lines = fileContent.trim().split('\n');
    const lossEntries: LossEntry[] = lines
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line));

    // Filter by step range if specified
    let filteredEntries = lossEntries;
    if (startStep > 0 || endStep >= 0) {
      filteredEntries = lossEntries.filter((entry) => {
        const afterStart = entry.step >= startStep;
        const beforeEnd = endStep < 0 || entry.step <= endStep;
        return afterStart && beforeEnd;
      });
    }

    if (filteredEntries.length === 0) {
      return NextResponse.json({
        steps: [],
        losses: [],
        ema90: [],
        ema95: [],
        ema99: [],
      });
    }

    // Extract steps and losses
    const steps = filteredEntries.map((entry) => entry.step);
    const losses = filteredEntries.map((entry) => entry.loss);

    // Calculate EMAs on full dataset before downsampling for accuracy
    // Using smoothing factors: 0.9 (light), 0.95 (medium), 0.99 (heavy)
    const emas = calculateMultipleEMAs(losses, [0.9, 0.95, 0.99]);

    // Downsample if data is larger than requested resolution
    if (steps.length > resolution) {
      // Prepare data for downsampling
      const rawData = prepareDataForDownsampling(steps, losses);
      const ema90Data = prepareDataForDownsampling(steps, emas[0.9]);
      const ema95Data = prepareDataForDownsampling(steps, emas[0.95]);
      const ema99Data = prepareDataForDownsampling(steps, emas[0.99]);

      // Downsample all series
      const downsampledRaw = downsampleLTTB(rawData, resolution);
      const downsampledEma90 = downsampleLTTB(ema90Data, resolution);
      const downsampledEma95 = downsampleLTTB(ema95Data, resolution);
      const downsampledEma99 = downsampleLTTB(ema99Data, resolution);

      // Extract downsampled data
      const rawExtracted = extractDownsampledData(downsampledRaw);
      const ema90Extracted = extractDownsampledData(downsampledEma90);
      const ema95Extracted = extractDownsampledData(downsampledEma95);
      const ema99Extracted = extractDownsampledData(downsampledEma99);

      return NextResponse.json({
        steps: rawExtracted.steps,
        losses: rawExtracted.values,
        ema90: ema90Extracted.values,
        ema95: ema95Extracted.values,
        ema99: ema99Extracted.values,
        totalPoints: lossEntries.length,
        downsampled: true,
      });
    } else {
      // Return full data if it's already small enough
      return NextResponse.json({
        steps,
        losses,
        ema90: emas[0.9],
        ema95: emas[0.95],
        ema99: emas[0.99],
        totalPoints: lossEntries.length,
        downsampled: false,
      });
    }
  } catch (error) {
    console.error('Error reading loss data file:', error);
    return NextResponse.json(
      { error: 'Error reading loss data file' },
      { status: 500 }
    );
  }
}
