'use client';
import { useState, type FC, useMemo } from 'react';
import type { Dataset, ColumnType } from '@/lib/types';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';
import { View } from '../layout/dashboard-layout';
import { CorrelationHeatmap } from './visualize/correlation-heatmap';
import { DistributionChart } from './visualize/distribution-chart';
import { PairPlot } from './visualize/pair-plot';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '../ui/label';

interface VisualizeViewProps {
  dataset: Dataset;
  onNavigate: (view: View) => void;
}

type SingleVarChart = 'histogram' | 'bar';
type MultiVarChart = 'scatter';
type ChartType = SingleVarChart | MultiVarChart;

const getCompatibleChartTypes = (
  columnType: ColumnType
): ChartType[] => {
  switch (columnType) {
    case 'numerical':
      return ['histogram', 'scatter'];
    case 'categorical':
    case 'boolean':
    case 'date':
      return ['bar'];
    default:
      return [];
  }
};

export const VisualizeView: FC<VisualizeViewProps> = ({ dataset, onNavigate }) => {
  const [selectedColumn1, setSelectedColumn1] = useState<string>(dataset.features.all[0]);
  const [selectedColumn2, setSelectedColumn2] = useState<string | null>(null);

  const numericalFeatures = useMemo(() => Object.entries(dataset.analysis.columnTypes)
    .filter(([, type]) => type === 'numerical')
    .map(([name]) => name), [dataset]);
  
  const compatibleChartTypes = useMemo(() => {
    return getCompatibleChartTypes(dataset.analysis.columnTypes[selectedColumn1]);
  }, [selectedColumn1, dataset.analysis.columnTypes]);
  
  const [selectedChart, setSelectedChart] = useState<ChartType>(compatibleChartTypes[0]);

  const handleColumn1Change = (column: string) => {
    setSelectedColumn1(column);
    const newCompatibleCharts = getCompatibleChartTypes(dataset.analysis.columnTypes[column]);
    setSelectedChart(newCompatibleCharts[0]);
    if (newCompatibleCharts[0] !== 'scatter') {
        setSelectedColumn2(null);
    } else {
        const otherNumFeatures = numericalFeatures.filter(f => f !== column);
        if (otherNumFeatures.length > 0) {
            setSelectedColumn2(otherNumFeatures[0]);
        } else {
            setSelectedColumn2(null);
        }
    }
  };

  const handleChartChange = (chart: ChartType) => {
    setSelectedChart(chart);
    if (chart !== 'scatter') {
        setSelectedColumn2(null);
    } else {
        if (!selectedColumn2) {
             const otherNumFeatures = numericalFeatures.filter(f => f !== selectedColumn1);
            if (otherNumFeatures.length > 0) {
                setSelectedColumn2(otherNumFeatures[0]);
            }
        }
    }
  }

  const renderChartExplorer = () => {
    if (!selectedColumn1 || !selectedChart) {
      return (
        <div className="flex items-center justify-center h-full bg-muted/50 rounded-b-lg">
            <p className="text-muted-foreground">Select a column to generate a chart.</p>
        </div>
      )
    }

    if (selectedChart === 'scatter') {
        if (!selectedColumn2) {
            return (
                <div className="flex items-center justify-center h-full bg-muted/50 rounded-b-lg">
                    <p className="text-muted-foreground">Please select a second numerical column for the scatter plot.</p>
                </div>
            )
        }
        return <PairPlot dataset={dataset} xColumn={selectedColumn1} yColumn={selectedColumn2} />;
    }

    return <DistributionChart 
        dataset={dataset} 
        column={selectedColumn1} 
        chartType={selectedChart as 'histogram' | 'bar'} 
    />;
  }

  return (
    <div className="space-y-6">
      <ViewHeader
        title="Automated Visualization Suite"
        description="Key visualizations generated from your data to uncover patterns and insights."
      />
      
      <div className="space-y-2">
        <h3 className='text-xl font-semibold'>Automated Insights</h3>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
            <Card>
            <CardHeader>
                <CardTitle>Target Variable Distribution</CardTitle>
                <CardDescription>Distribution of outcomes for '{dataset.features.target}'.</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
                {dataset.features.target && (
                  <DistributionChart 
                    dataset={dataset} 
                    column={dataset.features.target} 
                    chartType={dataset.analysis.columnTypes[dataset.features.target] === 'numerical' ? 'histogram' : 'bar'} 
                  />
                )}
            </CardContent>
            </Card>

            <Card>
            <CardHeader>
                <CardTitle>Correlation Heatmap</CardTitle>
                <CardDescription>Relationship between numerical features.</CardDescription>
            </CardHeader>
            <CardContent>
                <CorrelationHeatmap dataset={dataset} />
            </CardContent>
            </Card>
        </div>
      </div>

      <div className="space-y-2 pt-6">
        <h3 className='text-xl font-semibold'>Chart Explorer</h3>
        <Card>
            <CardHeader>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                     <div>
                        <Label htmlFor="column-select-1">Select Column (X-axis)</Label>
                        <Select value={selectedColumn1} onValueChange={handleColumn1Change}>
                            <SelectTrigger id="column-select-1">
                                <SelectValue placeholder="Select a column" />
                            </SelectTrigger>
                            <SelectContent>
                                {dataset.features.all.map(col => (
                                    <SelectItem key={col} value={col}>{col}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                     </div>
                     <div>
                        <Label htmlFor="chart-select">Select Chart Type</Label>
                        <Select 
                            value={selectedChart} 
                            onValueChange={(val) => handleChartChange(val as ChartType)}
                        >
                            <SelectTrigger id="chart-select">
                                <SelectValue placeholder="Select a chart type" />
                            </SelectTrigger>
                            <SelectContent>
                                {compatibleChartTypes.map(chart => (
                                    <SelectItem key={chart} value={chart}>{chart.charAt(0).toUpperCase() + chart.slice(1)}</SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                     </div>
                     {selectedChart === 'scatter' && (
                        <div>
                            <Label htmlFor="column-select-2">Select Column (Y-axis)</Label>
                             <Select value={selectedColumn2 ?? ''} onValueChange={setSelectedColumn2}>
                                <SelectTrigger id="column-select-2">
                                    <SelectValue placeholder="Select a column" />
                                </SelectTrigger>
                                <SelectContent>
                                    {numericalFeatures.filter(f => f !== selectedColumn1).map(col => (
                                        <SelectItem key={col} value={col}>{col}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                     )}
                </div>
            </CardHeader>
            <CardContent className='h-[400px]'>
                {renderChartExplorer()}
            </CardContent>
        </Card>
      </div>

      <div className="flex justify-end mt-6">
        <Button size="lg" onClick={() => onNavigate('features')}>
          Next: Select Features <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
