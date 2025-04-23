"use client"

import type React from "react"

import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { X } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface SettingsPanelProps {
  settings: {
    fontSize: number
    tabSize: number
    wordWrap: boolean
    minimap: boolean
    lineNumbers: boolean
    theme: string
  }
  setSettings: React.Dispatch<
    React.SetStateAction<{
      fontSize: number
      tabSize: number
      wordWrap: boolean
      minimap: boolean
      lineNumbers: boolean
      theme: string
    }>
  >
  onClose: () => void
}

export default function SettingsPanel({ settings, setSettings, onClose }: SettingsPanelProps) {
  return (
    <div className="h-full overflow-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium">Editor Settings</h3>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="fontSize">Font Size: {settings.fontSize}px</Label>
          <Slider
            id="fontSize"
            min={10}
            max={24}
            step={1}
            value={[settings.fontSize]}
            onValueChange={(value) => setSettings({ ...settings, fontSize: value[0] })}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="tabSize">Tab Size: {settings.tabSize}</Label>
          <Slider
            id="tabSize"
            min={2}
            max={8}
            step={2}
            value={[settings.tabSize]}
            onValueChange={(value) => setSettings({ ...settings, tabSize: value[0] })}
            className="w-full"
          />
        </div>

        <div className="space-y-4">
          <Label>Theme</Label>
          <Select value={settings.theme} onValueChange={(value) => setSettings({ ...settings, theme: value })}>
            <SelectTrigger className="w-full bg-[#1e2736] border-gray-700">
              <SelectValue placeholder="Select theme" />
            </SelectTrigger>
            <SelectContent className="bg-[#1a222e] border-gray-700">
              <SelectItem value="vs-dark">Dark</SelectItem>
              <SelectItem value="vs-light">Light</SelectItem>
              <SelectItem value="hc-black">High Contrast Dark</SelectItem>
              <SelectItem value="hc-light">High Contrast Light</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="wordWrap">Word Wrap</Label>
          <Switch
            id="wordWrap"
            checked={settings.wordWrap}
            onCheckedChange={(checked) => setSettings({ ...settings, wordWrap: checked })}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="minimap">Minimap</Label>
          <Switch
            id="minimap"
            checked={settings.minimap}
            onCheckedChange={(checked) => setSettings({ ...settings, minimap: checked })}
          />
        </div>

        <div className="flex items-center justify-between">
          <Label htmlFor="lineNumbers">Line Numbers</Label>
          <Switch
            id="lineNumbers"
            checked={settings.lineNumbers}
            onCheckedChange={(checked) => setSettings({ ...settings, lineNumbers: checked })}
          />
        </div>

        <Button
          className="w-full mt-4"
          variant="outline"
          onClick={() => {
            setSettings({
              fontSize: 14,
              tabSize: 4,
              wordWrap: true,
              minimap: false,
              lineNumbers: true,
              theme: "vs-dark",
            })
          }}
        >
          Reset to Defaults
        </Button>
      </div>
    </div>
  )
}
