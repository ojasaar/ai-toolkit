/**
 * Data point interface for downsampling
 */
export interface DataPoint {
  x: number;
  y: number;
}

/**
 * Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm
 * Preserves the visual shape of the data while reducing the number of points
 *
 * Reference: https://github.com/sveinn-steinarsson/flot-downsample
 *
 * @param data - Array of data points {x, y}
 * @param threshold - Target number of points to keep
 * @returns Downsampled array of data points
 */
export function downsampleLTTB(
  data: DataPoint[],
  threshold: number
): DataPoint[] {
  const dataLength = data.length;

  if (threshold >= dataLength || threshold <= 2) {
    return data; // Nothing to do
  }

  const sampled: DataPoint[] = new Array(threshold);

  // Bucket size. Leave room for start and end data points
  const every = (dataLength - 2) / (threshold - 2);

  let a = 0; // Initially a is the first point in the triangle
  sampled[0] = data[a]; // Always add the first point

  for (let i = 0; i < threshold - 2; i++) {
    // Calculate point average for next bucket (containing c)
    let avgX = 0;
    let avgY = 0;
    let avgRangeStart = Math.floor((i + 1) * every) + 1;
    let avgRangeEnd = Math.floor((i + 2) * every) + 1;
    avgRangeEnd = avgRangeEnd < dataLength ? avgRangeEnd : dataLength;

    const avgRangeLength = avgRangeEnd - avgRangeStart;

    for (; avgRangeStart < avgRangeEnd; avgRangeStart++) {
      avgX += data[avgRangeStart].x;
      avgY += data[avgRangeStart].y;
    }
    avgX /= avgRangeLength;
    avgY /= avgRangeLength;

    // Get the range for this bucket
    let rangeOffs = Math.floor(i * every) + 1;
    const rangeTo = Math.floor((i + 1) * every) + 1;

    // Point a
    const pointAX = data[a].x;
    const pointAY = data[a].y;

    let maxArea = -1;
    let maxAreaPoint = rangeOffs;

    for (; rangeOffs < rangeTo; rangeOffs++) {
      // Calculate triangle area over three buckets
      const area =
        Math.abs(
          (pointAX - avgX) * (data[rangeOffs].y - pointAY) -
            (pointAX - data[rangeOffs].x) * (avgY - pointAY)
        ) * 0.5;

      if (area > maxArea) {
        maxArea = area;
        maxAreaPoint = rangeOffs;
      }
    }

    sampled[i + 1] = data[maxAreaPoint]; // Pick this point from the bucket
    a = maxAreaPoint; // This point is the next a (chosen point)
  }

  sampled[threshold - 1] = data[dataLength - 1]; // Always add the last point

  return sampled;
}

/**
 * Simple helper to convert parallel arrays to DataPoint format
 */
export function prepareDataForDownsampling(
  steps: number[],
  values: number[]
): DataPoint[] {
  return steps.map((step, i) => ({ x: step, y: values[i] }));
}

/**
 * Simple helper to convert DataPoint array back to parallel arrays
 */
export function extractDownsampledData(
  points: DataPoint[]
): { steps: number[]; values: number[] } {
  return {
    steps: points.map((p) => p.x),
    values: points.map((p) => p.y),
  };
}
