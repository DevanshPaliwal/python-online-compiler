"use client"

interface InputPanelProps {
  input: string
  setInput: (input: string) => void
}

export default function InputPanel({ input, setInput }: InputPanelProps) {
  return (
    <div className="h-full w-full bg-[#1e2736] p-4 font-mono text-sm">
      <div className="mb-2 text-gray-400">Enter input for your program (will be passed as stdin):</div>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="h-[calc(100%-2rem)] w-full resize-none rounded border border-gray-700 bg-[#1a222e] p-3 font-mono text-sm text-white focus:border-blue-500 focus:outline-none"
        placeholder="Enter input here..."
      />
    </div>
  )
}
