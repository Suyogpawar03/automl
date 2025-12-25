'use client';

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import type { Dataset } from '@/lib/types';
import { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';

export function PairPlot({ dataset, xColumn, yColumn }: { dataset: Dataset, xColumn: string, yColumn: string }) {
  
  const chartConfig = useMemo(() => ({
    [yColumn]: {
        label: yColumn,
        color: "hsl(var(--primary))",
    },
  }), [yColumn]);

  const data = useMemo(() => dataset.data.map((row) => ({
    x: row[xColumn],
    y: row[yColumn],
  })), [dataset, xColumn, yColumn]);

  return (
    <ChartContainer config={chartConfig} className="h-full w-full">
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart
          margin={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20,
          }}
        >
          <CartesianGrid />
          <XAxis
            type="number"
            dataKey="x"
            name={xColumn}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yColumn}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
          />
          <ChartTooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={<ChartTooltipContent />}
          />
          <Scatter
            name={`${xColumn} vs ${yColumn}`}
            data={data}
            fill="hsl(var(--primary))"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  );
}
