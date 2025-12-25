'use server';

import { ai } from '@/ai/genkit';
import { z } from 'genkit';

const GenerateColumnDataTypesInputSchema = z.object({
  columnNames: z.array(z.string()),
  dataSample: z.string(),
});

export type GenerateColumnDataTypesInput = z.infer<
  typeof GenerateColumnDataTypesInputSchema
>;

const GenerateColumnDataTypesOutputSchema = z.object({
  columns: z.array(
    z.object({
      name: z.string(),
      type: z.enum([
        'string',
        'numerical',
        'categorical',
        'boolean',
        'date',
        'other',
      ]),
    })
  ),
});

export type GenerateColumnDataTypesOutput = z.infer<
  typeof GenerateColumnDataTypesOutputSchema
>;

export async function generateColumnDataTypes(input: {
  columnNames: string[];
  dataSample: Record<string, any>[];
}): Promise<GenerateColumnDataTypesOutput> {
  return generateColumnDataTypesFlow({
    columnNames: input.columnNames,
    dataSample: JSON.stringify(input.dataSample, null, 2),
  });
}

const prompt = ai.definePrompt({
  name: 'generateColumnDataTypesPrompt',
  input: { schema: GenerateColumnDataTypesInputSchema },
  output: { schema: GenerateColumnDataTypesOutputSchema },
  prompt: `
You are an expert data analyst.

Determine the data type of each column using only these values:
string, numerical, categorical, boolean, date, other.

Return the result strictly in this JSON format:

{
  "columns": [
    { "name": "column_name", "type": "numerical" }
  ]
}

Column names:
{{#each columnNames}}
- {{this}}
{{/each}}

Data sample:
{{{dataSample}}}
`,
});

const generateColumnDataTypesFlow = ai.defineFlow(
  {
    name: 'generateColumnDataTypesFlow',
    inputSchema: GenerateColumnDataTypesInputSchema,
    outputSchema: GenerateColumnDataTypesOutputSchema,
  },
  async input => {
    const { output } = await prompt(input);
    return output!;
  }
);
