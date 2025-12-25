'use client';

import { useMemo, type FC } from 'react';
import type { Dataset } from '@/lib/types';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts';

interface DistributionChartProps {
    dataset: Dataset;
    column: string;
    chartType: 'histogram' | 'bar';
}

const chartConfig = {
  count: {
    label: "Count",
    color: "hsl(var(--primary))",
  },
};

const calculateHistogram = (data: number[], bins = 10) => {
    const validData = data.filter(d => d !== null && !isNaN(d));
    if (validData.length === 0) return [];
    
    const min = Math.min(...validData);
    const max = Math.max(...validData);
    const range = max - min;
    
    if (range === 0) {
        return [{ range: `${min.toFixed(2)}`, count: validData.length }];
    }

    const binWidth = range / bins;
    
    const histogram: { range: string; count: number }[] = Array.from({ length: bins }, (_, i) => {
      const start = min + i * binWidth;
      const end = start + binWidth;
      return { range: `${start.toFixed(2)}-${end.toFixed(2)}`, count: 0 };
    });

    validData.forEach(value => {
        let binIndex = Math.floor((value - min) / binWidth);
        if (binIndex === bins) binIndex = bins - 1;
        if(histogram[binIndex]) {
            histogram[binIndex].count++;
        }
    });

    return histogram;
}

const calculateBarData = (data: (string | number | boolean)[]) => {
    const counts = data.reduce((acc, value) => {
        const key = String(value);
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    return Object.entries(counts).map(([name, count]) => ({name, count})).sort((a,b) => b.count - a.count);
}


export const DistributionChart: FC<DistributionChartProps> = ({ dataset, column, chartType }) => {

    const chartData = useMemo(() => {
        const columnData = dataset.data.map(row => row[column]).filter(v => v !== null && v !== undefined);
        
        switch (chartType) {
            case 'histogram':
                return calculateHistogram(columnData as number[]);
            case 'bar':
                return calculateBarData(columnData);
            default:
                return [];
        }
    }, [dataset, column, chartType]);

    if (!chartData || chartData.length === 0) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">Not enough data to render this chart.</p>
            </div>
        )
    }

    if (chartType === 'histogram' || chartType === 'bar') {
      const dataKey = chartType === 'histogram' ? 'range' : 'name';
      return (
        <ChartContainer config={chartConfig} className="h-full w-full">
            <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 20, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid vertical={false} />
                <XAxis
                dataKey={dataKey}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                />
                <YAxis tickLine={false} axisLine={false} tickMargin={8} />
                <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent />}
                />
                <Bar dataKey="count" fill="var(--color-count)" radius={4} />
            </BarChart>
            </ResponsiveContainer>
        </ChartContainer>
      );
    }
  
    return (
        <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground">Chart type '{chartType}' not yet implemented for this column.</p>
        </div>
    )
};
