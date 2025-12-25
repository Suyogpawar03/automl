"use client";

import { useState } from "react";
import {
  uploadCSV,
  preprocess,
  trainModel,
  runUnsupervised,
} from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");

  return (
    <main className="min-h-screen bg-gray-950 text-white p-8">
      <h1 className="text-3xl font-bold mb-6">AutoML Dashboard</h1>

      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-4"
      />

      <div className="flex gap-4">
        <button
          onClick={async () => {
            if (!file) return;
            setStatus("Uploading...");
            await uploadCSV(file);
            setStatus("Upload completed");
          }}
          className="px-4 py-2 bg-blue-600 rounded"
        >
          Upload
        </button>

        <button
          onClick={async () => {
            setStatus("Preprocessing...");
            await preprocess();
            setStatus("Preprocess done");
          }}
          className="px-4 py-2 bg-green-600 rounded"
        >
          Preprocess
        </button>

        <button
          onClick={async () => {
            setStatus("Training...");
            await trainModel();
            setStatus("Training completed");
          }}
          className="px-4 py-2 bg-purple-600 rounded"
        >
          Train
        </button>

        <button
          onClick={async () => {
            setStatus("Running clustering...");
            await runUnsupervised();
            setStatus("Unsupervised done");
          }}
          className="px-4 py-2 bg-yellow-600 rounded"
        >
          Unsupervised
        </button>
      </div>

      <p className="mt-6 text-green-400">{status}</p>
    </main>
  );
}
