import { type NextRequest, NextResponse } from "next/server"

// Piston API endpoint
const PISTON_API = "https://emkc.org/api/v2/piston/execute"

export async function POST(request: NextRequest) {
  try {
    const { code, input } = await request.json()

    if (!code) {
      return NextResponse.json({ error: "No code provided" }, { status: 400 })
    }

    // Prepare the request to the Piston API
    const pistonRequest = {
      language: "python",
      version: "3.10",
      files: [
        {
          name: "main.py",
          content: code,
        },
      ],
      stdin: input || "",
      args: [],
      compile_timeout: 10000,
      run_timeout: 5000,
      compile_memory_limit: -1,
      run_memory_limit: -1,
    }

    // Send the request to the Piston API
    const response = await fetch(PISTON_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(pistonRequest),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error("Piston API error:", errorText)
      return NextResponse.json({ error: "Failed to execute code" }, { status: 500 })
    }

    const data = await response.json()

    // Format the output
    let output = ""

    if (data.compile && data.compile.output) {
      output += `Compilation Output:\n${data.compile.output}\n\n`
    }

    if (data.run) {
      if (data.run.stdout) {
        output += data.run.stdout
      }

      if (data.run.stderr) {
        output += `\nError:\n${data.run.stderr}`
      }
    }

    return NextResponse.json({ output })
  } catch (error) {
    console.error("Server error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
