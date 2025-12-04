/**
 * Calculate Exponential Moving Average (EMA) for a series of values
 * EMA gives more weight to previous values based on the smoothing factor
 *
 * Formula: EMA[i] = alpha * EMA[i-1] + (1 - alpha) * value[i]
 *
 * @param values - Array of numeric values to smooth
 * @param alpha - Smoothing factor (0-1). Higher = more smoothing. Common: 0.9, 0.95, 0.99
 * @returns Array of EMA values (same length as input)
 */
export function calculateEMA(values: number[], alpha: number): number[] {
  if (values.length === 0) return [];
  if (alpha <= 0 || alpha >= 1) return values;

  const ema: number[] = new Array(values.length);

  // First EMA value is the first actual value
  ema[0] = values[0];

  // Calculate EMA for subsequent values
  // EMA[i] = alpha * EMA[i-1] + (1 - alpha) * value[i]
  for (let i = 1; i < values.length; i++) {
    ema[i] = alpha * ema[i - 1] + (1 - alpha) * values[i];
  }

  return ema;
}

/**
 * Calculate multiple EMAs at once for efficiency
 *
 * @param values - Array of numeric values to smooth
 * @param alphas - Array of smoothing factors to calculate EMAs for
 * @returns Object mapping smoothing factor to EMA array
 */
export function calculateMultipleEMAs(
  values: number[],
  alphas: number[]
): Record<number, number[]> {
  const result: Record<number, number[]> = {};

  for (const alpha of alphas) {
    result[alpha] = calculateEMA(values, alpha);
  }

  return result;
}
