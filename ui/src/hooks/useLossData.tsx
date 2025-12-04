'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';

export interface LossData {
  steps: number[];
  losses: number[];
  ema90: number[];
  ema95: number[];
  ema99: number[];
  totalPoints?: number;
  downsampled?: boolean;
}

export default function useLossData(
  jobID: string,
  reloadInterval: null | number = null
) {
  const [lossData, setLossData] = useState<LossData>({
    steps: [],
    losses: [],
    ema90: [],
    ema95: [],
    ema99: [],
  });
  const didInitialLoadRef = useRef(false);
  const [status, setStatus] = useState<
    'idle' | 'loading' | 'success' | 'error' | 'refreshing'
  >('idle');

  const refresh = () => {
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    apiClient
      .get(`/api/jobs/${jobID}/loss`)
      .then((res) => res.data)
      .then((data) => {
        setLossData({
          steps: data.steps || [],
          losses: data.losses || [],
          ema90: data.ema90 || [],
          ema95: data.ema95 || [],
          ema99: data.ema99 || [],
          totalPoints: data.totalPoints,
          downsampled: data.downsampled,
        });
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch((error) => {
        console.error('Error fetching loss data:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refresh();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refresh();
      }, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID]);

  return { lossData, setLossData, status, refresh };
}
