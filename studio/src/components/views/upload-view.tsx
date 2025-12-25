'use client';
import { useCallback, useState, type FC } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, File as FileIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

interface UploadViewProps {
  onDatasetUpload: (file: File) => void;
}

export const UploadView: FC<UploadViewProps> = ({ onDatasetUpload }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false,
  });

  const handleUpload = () => {
    if (file) {
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 95) {
            clearInterval(interval);
            return prev;
          }
          return prev + 5;
        });
      }, 100);

      // Simulate network delay and call parent handler
      setTimeout(() => {
          setUploadProgress(100);
          onDatasetUpload(file);
      }, 2000);
    }
  };

  const removeFile = () => {
    setFile(null);
    setUploadProgress(0);
  }

  return (
    <div className="flex flex-col items-center justify-center h-full w-full">
        <div className="w-full max-w-2xl text-center">
            <h1 className="text-4xl font-bold tracking-tight">ML Project Studio</h1>
            <p className="mt-4 text-lg text-muted-foreground">
                Upload your dataset to begin an analysis and visualization journey.
            </p>
        </div>

      <Card className="mt-10 w-full max-w-2xl">
        <CardContent className="p-6">
          {!file ? (
            <div
              {...getRootProps()}
              className={`flex flex-col items-center justify-center p-12 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                isDragActive ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'
              }`}
            >
              <input {...getInputProps()} />
              <UploadCloud className="w-16 h-16 text-muted-foreground" />
              <p className="mt-4 text-lg font-semibold">
                {isDragActive ? 'Drop the file here...' : 'Drag & drop a file here, or click to select'}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                Supports .csv, .xls, .xlsx files up to 50MB
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 border rounded-lg">
                <div className='flex items-center gap-3'>
                    <FileIcon className="w-8 h-8 text-primary" />
                    <div>
                        <p className="font-medium truncate">{file.name}</p>
                        <p className="text-sm text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                    </div>
                </div>
                <Button variant="ghost" size="icon" onClick={removeFile}>
                    <X className="w-5 h-5" />
                </Button>
              </div>

              {uploadProgress > 0 && <Progress value={uploadProgress} className="w-full" />}
              
              <Button onClick={handleUpload} disabled={uploadProgress > 0} className="w-full" size="lg">
                {uploadProgress > 0 && uploadProgress < 100 ? 'Analyzing...' : 'Start Analysis'}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
