---
name: Clinical Utilitarian
colors:
  surface: '#fdf7ff'
  surface-dim: '#ded8e0'
  surface-bright: '#fdf7ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f8f2fa'
  surface-container: '#f2ecf4'
  surface-container-high: '#ece6ee'
  surface-container-highest: '#e6e0e9'
  on-surface: '#1d1b20'
  on-surface-variant: '#494551'
  inverse-surface: '#322f35'
  inverse-on-surface: '#f5eff7'
  outline: '#7a7582'
  outline-variant: '#cbc4d2'
  surface-tint: '#6750a4'
  primary: '#4f378a'
  on-primary: '#ffffff'
  primary-container: '#6750a4'
  on-primary-container: '#e0d2ff'
  inverse-primary: '#cfbcff'
  secondary: '#63597c'
  on-secondary: '#ffffff'
  secondary-container: '#e1d4fd'
  on-secondary-container: '#645a7d'
  tertiary: '#765b00'
  on-tertiary: '#ffffff'
  tertiary-container: '#c9a74d'
  on-tertiary-container: '#503d00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#cfbcff'
  on-primary-fixed: '#22005d'
  on-primary-fixed-variant: '#4f378a'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#cdc0e9'
  on-secondary-fixed: '#1f1635'
  on-secondary-fixed-variant: '#4b4263'
  tertiary-fixed: '#ffdf93'
  tertiary-fixed-dim: '#e7c365'
  on-tertiary-fixed: '#241a00'
  on-tertiary-fixed-variant: '#594400'
  background: '#fdf7ff'
  on-background: '#1d1b20'
  surface-variant: '#e6e0e9'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '800'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '800'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  data-lg:
    fontFamily: JetBrains Mono
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1'
  body-md:
    fontFamily: JetBrains Mono
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 16px
  margin-mobile: 16px
  margin-desktop: 32px
---

## Brand & Style
The design system is engineered for high-stakes pediatric environments where cognitive load must be minimized to ensure patient safety. The personality is clinical, disciplined, and hyper-functional. It avoids all decorative flourishes in favor of brutal clarity and information density.

The aesthetic follows a **Restrained Minimalist** approach with a **Bento-box** structural logic. Every element is defined by its utility; if a component does not serve a data-reporting or navigational purpose, it is removed. Visual hierarchy is established through structural alignment and typographic weight rather than depth or color. The emotional response should be one of professional calm, precision, and absolute reliability.

## Colors
The palette is rooted in the OKLCH color space to ensure perceptual uniformity and clinical accuracy. 

- **Base Neutral**: A soft, clinical off-white used for all primary surfaces to reduce eye strain during long shifts. 
- **Text Neutral**: A deep slate used for all primary communication, ensuring high legibility without the harshness of pure black.
- **Primary Accent**: A precise Clinical Teal used for active states, primary actions, and positive status indicators.
- **Alert**: A muted Coral/Amber reserved strictly for warnings, overdue vitals, or capacity thresholds.

**Strict Implementation Rule**: Pure black (`#000000`) and pure white (`#FFFFFF`) are prohibited. All surfaces and strokes must use the defined OKLCH tokens to maintain the "tinted" clinical atmosphere.

## Typography
Typography is the primary driver of hierarchy. This design system utilizes a dual-font strategy: **Inter** for structural UI and headers to provide a modern, geometric feel, and **JetBrains Mono** for all patient data, counts, and telemetry logs.

- **Tabular Figures**: All data-driven numbers must utilize `font-variant-numeric: tabular-nums` to ensure columns align perfectly in dashboards.
- **Variable Weights**: Integers in counters should utilize weights between 700 and 800 to draw immediate attention.
- **Tracking**: Headlines use tight tracking (-0.01em to -0.02em) to appear as solid "blocks" of information, reinforcing the brutalist aesthetic.

## Layout & Spacing
The layout follows an **Asymmetric Bento-box** model. Content is organized into modular rectangular cells that grow or shrink based on priority.

- **Grid**: A 12-column fluid grid on desktop, collapsing to a single-column stack on mobile.
- **Borders**: Elements are separated by 1px solid tinted borders (`border_color_oklch`). No gaps are used between cards; they should share borders to create a unified technical "instrument panel" look.
- **Scaling**: On mobile, the most critical "Patient Count" telemetry moves to the top of the viewport, with secondary logs tucked into expandable drawers.

## Elevation & Depth
This design system rejects depth. There are **no shadows, no gradients, and no glassmorphism**. 

Hierarchy is achieved solely through **Tonal Layering** and **Line Weight**:
- **Level 0 (Background)**: The base neutral surface.
- **Level 1 (Cells)**: Defined by 1px solid borders.
- **Level 2 (Active/Interaction)**: Represented by a fill of the `primary_color_oklch` or a high-contrast weight shift in typography.

If an element needs to "pop" (such as a modal or alert), it should use a high-contrast stroke or a solid color fill, rather than an elevation shadow.

## Shapes
In line with the clinical and utilitarian aesthetic, this design system uses **Sharp (0px)** corners for all UI elements. 

The use of 90-degree angles reinforces the sense of a rigid, scientific instrument. This applies to buttons, input fields, cards, and navigation tabs. The only exception to the "no curves" rule is if a specific medical icon requires it for anatomical accuracy.

## Components
- **Telemetry Counters**: Large-scale blocks using JetBrains Mono (700-800 weight). The label sits at the top-left in `label-caps`. Real-time updates should "flash" the border color briefly rather than using a fade animation.
- **Buttons**: Rectangular, 1px border, uppercase Inter text. The primary state is a solid teal fill with the background-neutral color for text.
- **Data Tables**: Zero-padding on the outer edges to align with the bento-box borders. Row separators are 1px solid.
- **Status Chips**: Small rectangular blocks with a subtle background tint (OKLCH 90% L) and a corresponding 2px left-border "indicator" stripe.
- **Input Fields**: Labeled with `label-caps` above the field. Bottom-border only (2px) when focused to maintain the modernist, architectural feel.