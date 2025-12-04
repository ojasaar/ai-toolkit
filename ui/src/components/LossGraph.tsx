'use client';

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import useLossData from '@/hooks/useLossData';

interface LossGraphProps {
  jobID: string;
  refreshInterval?: number;
}

export default function LossGraph({
  jobID,
  refreshInterval = 5000,
}: LossGraphProps) {
  const { lossData, status } = useLossData(jobID, refreshInterval);
  const [showEma90, setShowEma90] = useState(false);
  const [showEma95, setShowEma95] = useState(true);
  const [showEma99, setShowEma99] = useState(false);

  // Transform data for Recharts
  const chartData = lossData.steps.map((step, index) => ({
    step,
    loss: lossData.losses[index],
    ema90: lossData.ema90[index],
    ema95: lossData.ema95[index],
    ema99: lossData.ema99[index],
  }));

  if (status === 'error') {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <p className="text-red-400">Error loading loss data</p>
      </div>
    );
  }

  if (lossData.steps.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-2">Training Loss</h3>
        <p className="text-gray-400 text-sm">
          No loss data available yet. Data will appear once training starts logging.
        </p>
      </div>
    );
  }

  const latestLoss = lossData.losses[lossData.losses.length - 1];
  const latestStep = lossData.steps[lossData.steps.length - 1];

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Training Loss</h3>
          <p className="text-sm text-gray-400">
            Step {latestStep.toLocaleString()} | Loss: {latestLoss.toFixed(6)}
            {lossData.downsampled && (
              <span className="ml-2 text-xs text-gray-500">
                (showing {lossData.steps.length.toLocaleString()} of{' '}
                {lossData.totalPoints?.toLocaleString()} points)
              </span>
            )}
          </p>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setShowEma90(!showEma90)}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              showEma90
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            EMA 0.9
          </button>
          <button
            onClick={() => setShowEma95(!showEma95)}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              showEma95
                ? 'bg-green-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            EMA 0.95
          </button>
          <button
            onClick={() => setShowEma99(!showEma99)}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              showEma99
                ? 'bg-purple-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            EMA 0.99
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="step"
            stroke="#9CA3AF"
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
            label={{
              value: 'Step',
              position: 'insideBottom',
              offset: -5,
              fill: '#9CA3AF',
            }}
          />
          <YAxis
            stroke="#9CA3AF"
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
            label={{
              value: 'Loss',
              angle: -90,
              position: 'insideLeft',
              fill: '#9CA3AF',
            }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '0.375rem',
              color: '#F3F4F6',
            }}
            labelStyle={{ color: '#F3F4F6' }}
            formatter={(value: number) => value.toFixed(6)}
          />
          <Legend
            wrapperStyle={{ color: '#9CA3AF' }}
            iconType="line"
          />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#EF4444"
            strokeWidth={1}
            dot={false}
            name="Raw Loss"
            isAnimationActive={false}
          />
          {showEma90 && (
            <Line
              type="monotone"
              dataKey="ema90"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
              name="EMA 0.9"
              isAnimationActive={false}
            />
          )}
          {showEma95 && (
            <Line
              type="monotone"
              dataKey="ema95"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
              name="EMA 0.95"
              isAnimationActive={false}
            />
          )}
          {showEma99 && (
            <Line
              type="monotone"
              dataKey="ema99"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={false}
              name="EMA 0.99"
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
