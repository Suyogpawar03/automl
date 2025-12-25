const API_URL = process.env.NEXT_PUBLIC_API_URL!;

export const uploadCSV = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  return res.json();
};

export const preprocess = async () =>
  fetch(`${API_URL}/preprocess`, { method: "POST" }).then(res => res.json());

export const trainModel = async () =>
  fetch(`${API_URL}/train`, { method: "POST" }).then(res => res.json());

export const runUnsupervised = async () =>
  fetch(`${API_URL}/unsupervised`, { method: "POST" }).then(res => res.json());
