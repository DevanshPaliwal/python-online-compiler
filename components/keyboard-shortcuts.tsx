"use client"

export default function KeyboardShortcuts() {
  const shortcuts = [
    { keys: ["Ctrl", "Enter"], description: "Run code" },
    { keys: ["Ctrl", "S"], description: "Save/download code" },
    { keys: ["Ctrl", "F"], description: "Find in code" },
    { keys: ["Ctrl", "H"], description: "Replace in code" },
    { keys: ["Ctrl", "/"], description: "Toggle comment" },
    { keys: ["Ctrl", "Z"], description: "Undo" },
    { keys: ["Ctrl", "Y"], description: "Redo" },
    { keys: ["Ctrl", "A"], description: "Select all" },
    { keys: ["Alt", "Up/Down"], description: "Move line up/down" },
    { keys: ["Ctrl", "D"], description: "Add selection to next find match" },
    { keys: ["F11"], description: "Toggle fullscreen" },
    { keys: ["Ctrl", "V"], description: "Paste from clipboard" },
  ]

  return (
    <div className="p-2">
      <h3 className="mb-4 text-lg font-medium">Keyboard Shortcuts</h3>
      <div className="space-y-2">
        {shortcuts.map((shortcut, index) => (
          <div key={index} className="flex items-center justify-between py-1">
            <span>{shortcut.description}</span>
            <div className="flex items-center gap-1">
              {shortcut.keys.map((key, keyIndex) => (
                <span key={keyIndex} className="rounded bg-gray-700 px-2 py-1 text-xs font-medium text-white">
                  {key}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs text-gray-400">Note: On Mac, use Cmd instead of Ctrl for most shortcuts.</p>
    </div>
  )
}
