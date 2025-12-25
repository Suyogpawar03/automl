export const parseCSV = (file: File): Promise<{ data: Record<string, any>[], columns: string[] }> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (event.target && typeof event.target.result === 'string') {
                const text = event.target.result;
                const lines = text.split(/\r\n|\n/);
                const headers = lines[0].split(',').map(h => h.trim());
                const data = [];
                for (let i = 1; i < lines.length; i++) {
                    if (!lines[i]) continue;
                    const values = lines[i].split(',');
                    const row: Record<string, any> = {};
                    for (let j = 0; j < headers.length; j++) {
                        const value = values[j]?.trim();
                        // Attempt to convert to number if possible
                        const numValue = Number(value);
                        row[headers[j]] = isNaN(numValue) || value === '' ? value : numValue;
                    }
                    data.push(row);
                }
                resolve({ data, columns: headers });
            } else {
                reject(new Error('Failed to read file.'));
            }
        };
        reader.onerror = () => {
            reject(new Error('Error reading file.'));
        };
        reader.readAsText(file);
    });
};
