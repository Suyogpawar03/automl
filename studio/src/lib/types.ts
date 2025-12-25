export type ColumnType = 'string' | 'numerical' | 'categorical' | 'boolean' | 'date' | 'other';

export interface Dataset {
  name: string;
  size: number;
  rowsCount: number;
  columnsCount: number;
  data: Record<string, any>[];
  preview: Record<string, any>[];
  analysis: {
    columnTypes: Record<string, ColumnType>;
    dataQuality: { 
      title: string;
      value: string;
      description: string;
    }[];
    statistics: Record<string, any>;
  };
  features: {
    all: string[];
    target: string | null;
    inputs: string[];
  };
}
