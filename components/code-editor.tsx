"use client"

import type React from "react"

import { useEffect } from "react"
import Editor from "@monaco-editor/react"

interface CodeEditorProps {
  code: string
  setCode: (code: string) => void
  settings: {
    fontSize: number
    tabSize: number
    wordWrap: boolean
    minimap: boolean
    lineNumbers: boolean
    theme: string
  }
  editorRef: React.MutableRefObject<any>
}

export default function CodeEditor({ code, setCode, settings, editorRef }: CodeEditorProps) {
  const handleEditorDidMount = (editor: any) => {
    editorRef.current = editor

    // Set editor options for better appearance
    editor.updateOptions({
      padding: { top: 10 },
      lineHeight: 22,
      fontFamily: "'Fira Code', monospace",
      renderLineHighlight: "all",
      scrollBeyondLastLine: false,
    })

    // Add custom commands
    editor.addAction({
      id: "run-code",
      label: "Run Code",
      keybindings: [window.monaco.KeyMod.CtrlCmd | window.monaco.KeyCode.Enter],
      run: () => {
        // This will be handled by the global event listener
      },
    })

    // Explicitly focus the editor to ensure it can receive clipboard events
    setTimeout(() => {
      editor.focus()
    }, 100)
  }

  // Update editor settings when they change
  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.updateOptions({
        fontSize: settings.fontSize,
        tabSize: settings.tabSize,
        wordWrap: settings.wordWrap ? "on" : "off",
        minimap: { enabled: settings.minimap },
        lineNumbers: settings.lineNumbers ? "on" : "off",
      })
    }
  }, [settings, editorRef])

  return (
    <div className="h-full w-full">
      <Editor
        height="100%"
        defaultLanguage="python"
        value={code}
        onChange={(value) => setCode(value || "")}
        theme={settings.theme}
        options={{
          minimap: { enabled: settings.minimap },
          fontSize: settings.fontSize,
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: settings.tabSize,
          wordWrap: settings.wordWrap ? "on" : "off",
          lineNumbers: settings.lineNumbers ? "on" : "off",
          renderLineHighlight: "all",
          fontLigatures: true,
          cursorBlinking: "smooth",
          cursorSmoothCaretAnimation: "on",
          smoothScrolling: true,
          glyphMargin: false,
          folding: true,
          contextmenu: true,
          rulers: [],
          colorDecorators: true,
          bracketPairColorization: {
            enabled: true,
          },
          autoIndent: "full",
          formatOnPaste: true,
          formatOnType: true,
          copyWithSyntaxHighlighting: true,
          // Explicitly enable clipboard operations
          find: {
            addExtraSpaceOnTop: false,
          },
          // Ensure the editor is editable
          readOnly: false,
        }}
        onMount={handleEditorDidMount}
        beforeMount={(monaco) => {
          // Make monaco available globally for the editor
          window.monaco = monaco
        }}
      />
    </div>
  )
}

// Add monaco to the window type
declare global {
  interface Window {
    monaco: any
  }
}
