"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Maximize2,
  Share2,
  Play,
  Settings,
  Code,
  Terminal,
  History,
  Copy,
  Download,
  Keyboard,
  ClipboardPaste,
} from "lucide-react"
import CodeEditor from "@/components/code-editor"
import OutputPanel from "@/components/output-panel"
import LanguageSidebar from "@/components/language-sidebar"
import InputPanel from "@/components/input-panel"
import TemplatesDropdown from "@/components/templates-dropdown"
import SettingsPanel from "@/components/settings-panel"
import KeyboardShortcuts from "@/components/keyboard-shortcuts"
import { runCode } from "@/lib/code-runner"
import { useToast } from "@/components/ui/use-toast"
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog"

const DEFAULT_CODE = `# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
print("Try programiz.pro")`

export default function Home() {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [userInput, setUserInput] = useState("")
  const [output, setOutput] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [activeTab, setActiveTab] = useState("output")
  const [showSettings, setShowSettings] = useState(false)
  const [editorSettings, setEditorSettings] = useState({
    fontSize: 14,
    tabSize: 4,
    wordWrap: true,
    minimap: false,
    lineNumbers: true,
    theme: "vs-dark",
  })
  const [executionHistory, setExecutionHistory] = useState<{ timestamp: Date; code: string; output: string }[]>([])
  const { toast } = useToast()
  const editorRef = useRef<any>(null)

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + Enter to run code
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault()
        handleRunCode()
      }

      // Ctrl/Cmd + S to save code
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault()
        handleDownload()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [code])

  const handleRunCode = async () => {
    setIsRunning(true)
    setOutput("Running code...")

    try {
      const result = await runCode(code, userInput)
      setOutput(result)

      // Add to execution history
      setExecutionHistory((prev) => [
        { timestamp: new Date(), code, output: result },
        ...prev.slice(0, 9), // Keep only the last 10 executions
      ])
    } catch (error) {
      setOutput(`Error: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsRunning(false)
    }
  }

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`)
      })
    } else {
      document.exitFullscreen()
    }
    setIsFullscreen(!isFullscreen)
  }

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code)
    toast({
      title: "Code copied to clipboard",
      duration: 2000,
    })
  }

  const handlePasteCode = async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text) {
        setCode(text)
        toast({
          title: "Code pasted from clipboard",
          duration: 2000,
        })
      }
    } catch (error) {
      console.error("Failed to read clipboard:", error)
      toast({
        title: "Unable to paste from clipboard",
        description: "Please try using keyboard shortcut Ctrl+V or Cmd+V instead",
        variant: "destructive",
        duration: 3000,
      })
    }
  }

  const handleDownload = () => {
    const element = document.createElement("a")
    const file = new Blob([code], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = "main.py"
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    toast({
      title: "Code downloaded as main.py",
      duration: 2000,
    })
  }

  return (
    <div className="flex h-screen flex-col bg-[#1e2736] text-white">
      {/* Header */}
      <header className="flex h-16 items-center justify-between border-b border-gray-800 bg-[#1e2736] px-6">
        <div className="flex items-center gap-2">
          <div className="text-2xl font-bold text-white">
            <span className="text-white">P</span>rogramiz
          </div>
          <div className="text-sm text-gray-300">Python Online Compiler</div>
        </div>
        <div className="flex items-center gap-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:bg-gray-700 hover:text-white">
                <Keyboard className="h-5 w-5" />
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <KeyboardShortcuts />
            </DialogContent>
          </Dialog>
          <Button className="rounded-md border border-gray-600 bg-transparent px-4 py-2 text-white hover:bg-gray-700">
            Programiz PRO &rarr;
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Language Sidebar */}
        <LanguageSidebar />

        {/* Editor and Output */}
        <div className="flex flex-1 flex-col">
          {/* Editor Controls */}
          <div className="flex h-12 items-center justify-between border-b border-gray-800 bg-[#1e2736] px-4">
            <div className="flex items-center gap-2">
              <div className="rounded bg-[#1e2736] px-3 py-1 text-sm text-white">main.py</div>
              <TemplatesDropdown setCode={setCode} />
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
                onClick={toggleFullscreen}
                title="Fullscreen"
              >
                <Maximize2 className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
                onClick={() => setShowSettings(!showSettings)}
                title="Settings"
              >
                <Settings className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
                onClick={handleCopyCode}
                title="Copy Code"
              >
                <Copy className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
                onClick={handlePasteCode}
                title="Paste Code"
              >
                <ClipboardPaste className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-gray-400 hover:bg-gray-700 hover:text-white"
                onClick={handleDownload}
                title="Download Code"
              >
                <Download className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                className="flex items-center gap-1 text-gray-400 hover:bg-gray-700 hover:text-white"
              >
                <Share2 className="h-4 w-4" />
                <span>Share</span>
              </Button>
              <Button className="bg-blue-600 px-6 hover:bg-blue-700" onClick={handleRunCode} disabled={isRunning}>
                <Play className="mr-1 h-4 w-4" />
                Run
              </Button>
            </div>
          </div>

          {/* Editor and Output Container */}
          <div className="flex flex-1 overflow-hidden">
            {/* Settings Panel (conditionally rendered) */}
            {showSettings && (
              <div className="w-64 border-r border-gray-800 bg-[#1a222e] p-4">
                <SettingsPanel
                  settings={editorSettings}
                  setSettings={setEditorSettings}
                  onClose={() => setShowSettings(false)}
                />
              </div>
            )}

            {/* Code Editor */}
            <div
              className={`${showSettings ? "w-[calc(50%-16rem)]" : "w-1/2"} overflow-hidden border-r border-gray-800`}
            >
              <CodeEditor code={code} setCode={setCode} settings={editorSettings} editorRef={editorRef} />
            </div>

            {/* Output Panel */}
            <div className="flex w-1/2 flex-col">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
                <div className="flex h-12 items-center border-b border-gray-800 bg-[#1e2736] px-4">
                  <TabsList className="grid w-full max-w-md grid-cols-3 bg-[#1a222e]">
                    <TabsTrigger value="output" className="data-[state=active]:bg-[#1e2736]">
                      <Code className="mr-2 h-4 w-4" />
                      Output
                    </TabsTrigger>
                    <TabsTrigger value="input" className="data-[state=active]:bg-[#1e2736]">
                      <Terminal className="mr-2 h-4 w-4" />
                      Input
                    </TabsTrigger>
                    <TabsTrigger value="history" className="data-[state=active]:bg-[#1e2736]">
                      <History className="mr-2 h-4 w-4" />
                      History
                    </TabsTrigger>
                  </TabsList>
                  <div className="ml-auto">
                    {activeTab === "output" && (
                      <Button
                        variant="ghost"
                        className="text-sm text-gray-400 hover:bg-gray-700 hover:text-white"
                        onClick={() => setOutput("")}
                      >
                        Clear
                      </Button>
                    )}
                    {activeTab === "input" && (
                      <Button
                        variant="ghost"
                        className="text-sm text-gray-400 hover:bg-gray-700 hover:text-white"
                        onClick={() => setUserInput("")}
                      >
                        Clear
                      </Button>
                    )}
                  </div>
                </div>

                <TabsContent value="output" className="flex-1 m-0 p-0 data-[state=inactive]:hidden">
                  <OutputPanel output={output} />
                </TabsContent>

                <TabsContent value="input" className="flex-1 m-0 p-0 data-[state=inactive]:hidden">
                  <InputPanel input={userInput} setInput={setUserInput} />
                </TabsContent>

                <TabsContent value="history" className="flex-1 m-0 p-0 data-[state=inactive]:hidden">
                  <div className="h-full w-full overflow-auto bg-[#1e2736] p-4 text-sm text-white">
                    {executionHistory.length > 0 ? (
                      <div className="space-y-4">
                        {executionHistory.map((item, index) => (
                          <div key={index} className="rounded border border-gray-700 p-3">
                            <div className="mb-2 flex items-center justify-between">
                              <span className="text-xs text-gray-400">{item.timestamp.toLocaleString()}</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 text-xs"
                                onClick={() => setCode(item.code)}
                              >
                                Load
                              </Button>
                            </div>
                            <div className="max-h-20 overflow-hidden text-ellipsis whitespace-nowrap border-l-2 border-blue-500 pl-2">
                              <pre className="text-xs">
                                {item.code.slice(0, 100)}
                                {item.code.length > 100 ? "..." : ""}
                              </pre>
                            </div>
                            <div className="mt-2 max-h-20 overflow-hidden text-ellipsis border-l-2 border-green-500 pl-2">
                              <pre className="text-xs">
                                {item.output.slice(0, 100)}
                                {item.output.length > 100 ? "..." : ""}
                              </pre>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex h-full items-center justify-center text-gray-500">
                        No execution history yet. Run your code to see history.
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
