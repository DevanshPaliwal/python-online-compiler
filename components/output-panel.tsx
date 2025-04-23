"use client"

import { useEffect, useRef } from "react"

interface OutputPanelProps {
  output: string
}

export default function OutputPanel({ output }: OutputPanelProps) {
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  return (
    <div ref={outputRef} className="h-full w-full overflow-auto bg-[#1e2736] p-4 font-mono text-sm text-white">
      {output ? (
        <pre className="whitespace-pre-wrap">{output}</pre>
      ) : (
        <div className="flex h-full items-center justify-center text-gray-500">
          Run your code to see the output here
        </div>
      )}
    </div>
  )
}
