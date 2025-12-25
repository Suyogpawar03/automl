import type { FC } from 'react';
import { AlertCircle, ArrowRight, CheckCircle, FileText } from 'lucide-react';
import type { Dataset } from '@/lib/types';
import { ViewHeader } from '@/components/views/view-header';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { View } from '../layout/dashboard-layout';

interface AnalysisViewProps {
  dataset: Dataset;
  onNavigate: (view: View) => void;
}

const typeColors: { [key: string]: string } = {
  numerical: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  categorical: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  string: 'bg-green-500/20 text-green-400 border-green-500/30',
  boolean: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  date: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  other: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
};

export const AnalysisView: FC<AnalysisViewProps> = ({ dataset, onNavigate }) => {
  return (
    <div className="space-y-6">
      <ViewHeader
        title="Automated Data Analysis"
        description="Here's a summary of your dataset's characteristics and initial quality assessment."
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-primary" />
              Dataset Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">File Name</span>
                <span className="font-medium">{dataset.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Rows</span>
                <span className="font-medium">{dataset.rowsCount.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Columns</span>
                <span className="font-medium">{dataset.columnsCount.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">File Size</span>
                <span className="font-medium">{(dataset.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-primary" />
              Data Quality Report
            </CardTitle>
            <CardDescription>Initial validation and quality checks performed on your dataset.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {dataset.analysis.dataQuality.map((item) => (
                <div key={item.title} className="p-4 bg-card-foreground/5 rounded-lg">
                  <p className="text-sm text-muted-foreground">{item.title}</p>
                  <p className="text-2xl font-bold">{item.value}</p>
                  <p className="text-xs text-muted-foreground">{item.description}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Column Data Types</CardTitle>
            <CardDescription>Automatically inferred data types for each column.</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-72">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column Name</TableHead>
                    <TableHead>Inferred Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(dataset.analysis.columnTypes).map(([col, type]) => (
                    <TableRow key={col}>
                      <TableCell className="font-medium">{col}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={`${typeColors[type]}`}>{type}</Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Dataset Preview</CardTitle>
            <CardDescription>A glimpse of the first few rows of your data.</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-72">
              <Table>
                <TableHeader>
                  <TableRow>
                    {dataset.features.all.map((col) => (
                      <TableHead key={col}>{col}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {dataset.preview.map((row, i) => (
                    <TableRow key={i}>
                      {dataset.features.all.map((col) => (
                        <TableCell key={col}>{String(row[col])}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-end mt-6">
        <Button size="lg" onClick={() => onNavigate('visualize')}>
          Next: Visualize Data <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
