import { Ionicons } from "@expo/vector-icons";
import type { ComponentProps } from "react";

type IonName = ComponentProps<typeof Ionicons>["name"];

/** Distinct Ionicons + circle colors per wound type (blue–cream family + shape differentiation). */
export type WoundIconVisual = {
  name: IonName;
  circleBg: string;
  iconColor: string;
};

const DEFAULT: WoundIconVisual = {
  name: "medical-outline",
  circleBg: "#D4E4EF",
  iconColor: "#527A96",
};

export const WOUND_ICON_BY_TYPE: Record<string, WoundIconVisual> = {
  abrasion: {
    name: "layers-outline",
    circleBg: "#D4E4EF",
    iconColor: "#4A6B82",
  },
  bruise: {
    name: "color-filter-outline",
    circleBg: "#C5D8E8",
    iconColor: "#3D5A6B",
  },
  burn: {
    name: "flame-outline",
    circleBg: "#C9DDF0",
    iconColor: "#5E7FA3",
  },
  cut: {
    name: "bandage-outline",
    circleBg: "#D8E6F0",
    iconColor: "#4F7691",
  },
  ingrown_nail: {
    name: "hand-right-outline",
    circleBg: "#DEE8F0",
    iconColor: "#526F84",
  },
  laceration: {
    name: "warning-outline",
    circleBg: "#D4DEE8",
    iconColor: "#5C788E",
  },
  stab_wound: {
    name: "flash-outline",
    circleBg: "#D1DEE9",
    iconColor: "#3D5F78",
  },
};

export function getWoundIconVisual(woundType: string): WoundIconVisual {
  const key = woundType.toLowerCase().replace(/\s+/g, "_");
  return WOUND_ICON_BY_TYPE[key] ?? DEFAULT;
}
