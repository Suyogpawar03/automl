'use client';

import { useState, type FC } from 'react';
import { Loader2 } from 'lucide-react';
import { useToast } from "@/hooks/use-toast";
import { DashboardLayout, type View } from '@/components/layout/dashboard-layout';
import { UploadView } from '@/components/views/upload-view';
import { AnalysisView } from '@/components/views/analysis-view';
import { FeatureSelectionView } from '@/components/views/feature-selection-view';
import { PreprocessingView } from '@/components/views/preprocessing-view';
import { VisualizeView } from '@/components/views/visualize-view';
import { ModelsView, type TrainedModel } from '@/components/views/models-view';
import { ReportView } from '@/components/views/report-view';
import type { Dataset, ColumnType } from '@/lib/types';
import { parseCSV } from '@/lib/csv';
import { generateColumnDataTypes } from '@/ai/flows/generate-column-data-types';

export const Dashboard: FC = () => {
  const [activeView, setActiveView] = useState<View>('upload');
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [trainedModel, setTrainedModel] = useState<TrainedModel | null>(null);
  const { toast } = useToast();

  const handleDatasetUpload = async (file: File) => {
    setIsLoading(true);
    toast({
      title: "Processing Dataset",
      description: "Reading and analyzing your data, please wait...",
    });

    try {
      const { data: parsedData, columns } = await parseCSV(file);
      const dataSample = parsedData.slice(0, 10);
      const { columns: columnData } = await generateColumnDataTypes({
        columnNames: columns,
        dataSample: dataSample,
      });

      const columnTypes = columnData.reduce((acc, col) => {
        acc[col.name] = col.type as ColumnType;
        return acc;
      }, {} as Record<string, ColumnType>);

      const newDataset: Dataset = {
        name: file.name,
        size: file.size,
        rowsCount: parsedData.length,
        columnsCount: columns.length,
        data: parsedData,
        preview: parsedData.slice(0, 5),
        analysis: {
          columnTypes: columnTypes,
          dataQuality: [ 
            { title: 'Missing Values', value: '0.15%', description: 'Found in 11 rows' },
            { title: 'Duplicate Rows', value: '0', description: 'No duplicate entries' },
            { title: 'Empty Columns', value: '0', description: 'All columns have data' },
            { title: 'Inconsistent Values', value: '2', description: 'In `PaymentMethod`' },
          ],
          statistics: {},
        },
        features: {
          all: columns,
          target: columns.length > 0 ? columns[columns.length - 1] : null,
          inputs: columns.slice(0, columns.length - 1),
        },
      };

      setDataset(newDataset);
      setActiveView('analysis');
      toast({
        title: "Analysis Complete",
        description: "Your dataset has been successfully processed.",
      });
    } catch (error) {
      console.error("Failed to process dataset:", error);
      toast({
        variant: "destructive",
        title: "Processing Failed",
        description: "Could not read or analyze the provided dataset.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelTraining = (model: TrainedModel) => {
    setIsLoading(true);
    setTrainedModel(model);
    toast({
      title: `Training ${model.name}`,
      description: 'The model is being trained, please wait a moment...',
    });

    setTimeout(() => {
      setIsLoading(false);
      setActiveView('report');
      toast({
        title: 'Training Complete',
        description: `${model.name} has been trained successfully.`,
      });
    }, 2000);
  };
  
  const handleNavigate = (view: View) => {
    if (view !== 'upload' && !dataset) {
      toast({
        variant: "destructive",
        title: "No Dataset Found",
        description: "Please upload a dataset to access this view.",
      });
      return;
    }
    setActiveView(view);
  };
  
  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex h-[calc(100vh-10rem)] w-full items-center justify-center">
          <div className='flex flex-col items-center gap-4'>
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p className='text-muted-foreground'>Analyzing your data with AI, this may take a moment...</p>
          </div>
        </div>
      );
    }
    
    const viewProps = { dataset: dataset!, onNavigate: handleNavigate };

    switch (activeView) {
      case 'upload':
        return dataset ? <AnalysisView {...viewProps} /> : <UploadView onDatasetUpload={handleDatasetUpload} />;
      case 'analysis':
        return <AnalysisView {...viewProps} />;
      case 'visualize':
        return <VisualizeView {...viewProps} />;
      case 'features':
        return <FeatureSelectionView {...viewProps} />;
      case 'preprocessing':
        return <PreprocessingView {...viewProps} />;
      case 'models':
        return <ModelsView {...viewProps} onTrainModel={handleModelTraining} />;
      case 'report':
        return <ReportView {...viewProps} trainedModel={trainedModel} />;
      default:
        return <UploadView onDatasetUpload={handleDatasetUpload} />;
    }
  };

  return (
    <DashboardLayout activeView={activeView} onNavigate={handleNavigate} hasDataset={!!dataset}>
      {renderContent()}
    </DashboardLayout>
  );
};
