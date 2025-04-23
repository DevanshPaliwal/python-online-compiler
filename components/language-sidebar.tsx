"use client"

import { PiIcon as Python, FileCode2, Database, Code2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export default function LanguageSidebar() {
  return (
    <div className="flex w-14 flex-col items-center border-r border-gray-800 bg-[#1a222e] py-2">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="mb-2 rounded-md bg-blue-600 text-white hover:bg-blue-700">
              <Python className="h-5 w-5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p>Python</p>
          </TooltipContent>
        </Tooltip>

        <div className="mt-4 flex flex-col gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:bg-gray-700 hover:text-white">
                <FileCode2 className="h-5 w-5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>File Explorer</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:bg-gray-700 hover:text-white">
                <Database className="h-5 w-5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Database</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="text-gray-400 hover:bg-gray-700 hover:text-white">
                <Code2 className="h-5 w-5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Snippets</p>
            </TooltipContent>
          </Tooltip>
        </div>

        <div className="mt-4 flex flex-col gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex h-8 w-8 items-center justify-center rounded text-xs font-bold text-gray-400 hover:bg-gray-700 hover:text-white cursor-pointer">
                JS
              </div>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>JavaScript</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex h-8 w-8 items-center justify-center rounded text-xs font-bold text-gray-400 hover:bg-gray-700 hover:text-white cursor-pointer">
                TS
              </div>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>TypeScript</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex h-8 w-8 items-center justify-center rounded text-xs font-bold text-gray-400 hover:bg-gray-700 hover:text-white cursor-pointer">
                GO
              </div>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>Go</p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex h-8 w-8 items-center justify-center rounded text-xs font-bold text-gray-400 hover:bg-gray-700 hover:text-white cursor-pointer">
                PHP
              </div>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p>PHP</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>
    </div>
  )
}
