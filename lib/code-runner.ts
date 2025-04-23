export async function runCode(code: string, input = ""): Promise<string> {
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ code, input }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.error || "Failed to run code")
    }

    const data = await response.json()
    return data.output
  } catch (error) {
    console.error("Error running code:", error)
    throw error
  }
}
