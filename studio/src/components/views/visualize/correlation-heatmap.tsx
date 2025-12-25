'use client';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { Dataset } from '@/lib/types';
import { useMemo } from 'react';

// This is a simplified correlation calculation for demonstration.
// In a real application, this should be done in a backend or web worker.
const calculateCorrelationMatrix = (dataset: Dataset, features: string[]) => {
    const data = dataset.data;
    const n = data.length;
    
    const means: Record<string, number> = {};
    for (const feature of features) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += Number(data[i][feature]) || 0;
        }
        means[feature] = sum / n;
    }

    const stdDevs: Record<string, number> = {};
    for (const feature of features) {
        let sumSqDiff = 0;
        for (let i = 0; i < n; i++) {
            sumSqDiff += Math.pow((Number(data[i][feature]) || 0) - means[feature], 2);
        }
        stdDevs[feature] = Math.sqrt(sumSqDiff / (n -1));
    }
    
    const matrix: Record<string, Record<string, number>> = {};
    for (const feature1 of features) {
        matrix[feature1] = {};
        for (const feature2 of features) {
            if (feature1 === feature2) {
                matrix[feature1][feature2] = 1.0;
                continue;
            }
            if (matrix[feature2]?.[feature1] !== undefined) {
                 matrix[feature1][feature2] = matrix[feature2][feature1];
                 continue;
            }

            let covariance = 0;
            for (let i = 0; i < n; i++) {
                covariance += ((Number(data[i][feature1]) || 0) - means[feature1]) * ((Number(data[i][feature2]) || 0) - means[feature2]);
            }
            covariance /= (n - 1);
            
            const correlation = covariance / (stdDevs[feature1] * stdDevs[feature2]);
            matrix[feature1][feature2] = isNaN(correlation) ? 0 : correlation;
        }
    }
    return matrix;
};


const getCellColor = (value: number) => {
  const alpha = Math.abs(value);
  if (value > 0) {
    // Positive correlation: blue
    return `rgba(59, 130, 246, ${alpha})`;
  } else {
    // Negative correlation: red
    return `rgba(239, 68, 68, ${alpha})`;
  }
};

export function CorrelationHeatmap({ dataset }: { dataset: Dataset }) {
  const numericalFeatures = useMemo(() => Object.entries(dataset.analysis.columnTypes)
    .filter(([, type]) => type === 'numerical')
    .map(([name]) => name), [dataset]);

  const correlationMatrix = useMemo(() => calculateCorrelationMatrix(dataset, numericalFeatures), [dataset, numericalFeatures]);

  if (numericalFeatures.length < 2) {
    return <div className="flex items-center justify-center h-full"><p className="text-muted-foreground">Not enough numerical features for a correlation heatmap.</p></div>
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            {numericalFeatures.map((feature) => (
              <TableHead key={feature} className="text-center">
                {feature}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {numericalFeatures.map((rowFeature) => (
            <TableRow key={rowFeature}>
              <TableHead>{rowFeature}</TableHead>
              {numericalFeatures.map((colFeature) => {
                const correlationValue =
                  correlationMatrix[rowFeature]?.[colFeature];
                return (
                  <TableCell
                    key={`${rowFeature}-${colFeature}`}
                    className="text-center"
                    style={{
                      backgroundColor:
                        correlationValue !== undefined
                          ? getCellColor(correlationValue)
                          : 'transparent',
                      color:
                        correlationValue !== undefined &&
                        Math.abs(correlationValue) > 0.7
                          ? 'white'
                          : 'inherit',
                    }}
                  >
                    {correlationValue?.toFixed(2) ?? '-'}
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
