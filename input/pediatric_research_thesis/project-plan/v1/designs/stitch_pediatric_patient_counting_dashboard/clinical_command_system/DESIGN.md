---
name: Clinical Command System
colors:
  surface: '#f9f9fb'
  surface-dim: '#d9dadc'
  surface-bright: '#f9f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f5'
  surface-container: '#eeeef0'
  surface-container-high: '#e8e8ea'
  surface-container-highest: '#e2e2e4'
  on-surface: '#1a1c1d'
  on-surface-variant: '#414755'
  inverse-surface: '#2f3132'
  inverse-on-surface: '#f0f0f2'
  outline: '#717786'
  outline-variant: '#c1c6d7'
  surface-tint: '#005bc1'
  primary: '#0058bc'
  on-primary: '#ffffff'
  primary-container: '#0070eb'
  on-primary-container: '#fefcff'
  inverse-primary: '#adc6ff'
  secondary: '#5f5e60'
  on-secondary: '#ffffff'
  secondary-container: '#e2dfe1'
  on-secondary-container: '#636264'
  tertiary: '#ad2c00'
  on-tertiary: '#ffffff'
  tertiary-container: '#d83900'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a41'
  on-primary-fixed-variant: '#004493'
  secondary-fixed: '#e4e2e4'
  secondary-fixed-dim: '#c8c6c8'
  on-secondary-fixed: '#1b1b1d'
  on-secondary-fixed-variant: '#474649'
  tertiary-fixed: '#ffdbd1'
  tertiary-fixed-dim: '#ffb5a0'
  on-tertiary-fixed: '#3b0900'
  on-tertiary-fixed-variant: '#872000'
  background: '#f9f9fb'
  on-background: '#1a1c1d'
  surface-variant: '#e2e2e4'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: '0'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: '0'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: '0'
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  unit: 8px
  container-margin: 64px
  gutter: 24px
  section-gap: 48px
---

## Brand & Style

This design system is engineered for high-stakes clinical environments where cognitive load must be minimized to facilitate rapid, life-saving decision-making. The aesthetic is **Minimalist and High-End**, drawing inspiration from precision aerospace instrumentation and premium consumer hardware. 

The visual language communicates authority through "quiet" interfaces: expansive whitespace, hyper-refined typography, and a "Glassmorphism" material logic that suggests depth without clutter. The emotional response is one of **absolute calm, clinical precision, and effortless control**. By removing the traditional "boxiness" of medical software, the design system allows critical data to float as the primary hero of the experience.

## Colors

The palette is anchored by **Crisp White** (#FFFFFF) and **Space Gray** (#1C1C1E) to establish a sophisticated, high-contrast foundation. 

*   **Primary (Electric Blue):** Used for active states, primary actions, and "safe" progress. It provides a modern, technological feel.
*   **Secondary (Deep Space):** Reserved for sidebar navigation and high-level headers to ground the interface.
*   **Tertiary (International Orange):** A high-visibility accent strictly reserved for critical alerts, life-threatening vitals, and emergency overrides.
*   **Neutral (System Gray):** A range of cool grays used for backgrounds and subtle borders, ensuring the UI feels expansive and breathable.

## Typography

The design system utilizes **Inter** for its systematic, neutral, and highly legible characteristics. To achieve the Apple-inspired "Display" look, headlines utilize tight tracking (letter-spacing) and heavy weights, while labels utilize increased tracking for readability in fast-paced environments.

Scale is used as the primary tool for hierarchy. Large display sizes are used for patient names or critical metrics, while the "Label-LG" style (uppercase with tracking) is used for metadata and secondary identifiers to differentiate them from interactive body text.

## Layout & Spacing

The layout follows a **Fixed 12-Column Grid** on desktop to ensure data density is managed with mathematical precision. 

*   **Margins:** Extremely generous side margins (64px) create a "gallery" effect, centering the user's focus.
*   **Whitespace:** Avoid "filling" space. If a module has little content, allow the surrounding whitespace to remain empty to preserve the calm atmosphere.
*   **Breakpoints:** 
    *   *Desktop (1440px+):* 12 columns, 64px margins.
    *   *Tablet (1024px):* 8 columns, 32px margins.
    *   *Mobile (375px):* 4 columns, 16px margins. Headlines scale down by 20% to maintain legibility.

## Elevation & Depth

Depth is achieved through **Physical Materiality** rather than traditional drop shadows. 

1.  **Backdrop Blur:** Primary panels use a semi-transparent white or gray fill with a `backdrop-filter: blur(20px)`. This allows the colors of the background to bleed through softly, creating a sense of lightness.
2.  **Micro-Borders:** Instead of shadows, use 0.5px or 1px solid borders in a high-contrast color (e.g., White at 20% opacity) to define the edges of glass panels.
3.  **Tonal Stacking:** Higher-priority elements (like modals) should have a lighter background tint and a slightly stronger backdrop blur than the elements beneath them.

## Shapes

The shape language is defined by **Continuous Curvature**. 

While the standard `rounded-lg` is set to 32px, all primary containers and cards must use a minimum radius of **24px** to eliminate the "clinical" or "hostile" feel of sharp corners. Smaller components like buttons and input fields should follow a **Pill-shape** (fully rounded) or a minimum of 16px radius to maintain consistency with the larger containers. This softness acts as a psychological counterweight to the high-pressure nature of command center data.

## Components

*   **Glass Cards:** The core container. Features a 24px+ corner radius, 1px translucent border, and a 20px backdrop blur. No shadows.
*   **Primary Buttons:** Pill-shaped, high-contrast (Electric Blue or Space Gray background), with white text. Use a subtle inner-glow for a tactile, "etched" look.
*   **Status Chips:** Small, semi-transparent pills. For example, a "Critical" chip uses 10% International Orange fill with 100% opacity orange text.
*   **Input Fields:** Minimalist underlines or very soft, filled pills. Focus states are indicated by a 2px Electric Blue ring with a slight glow.
*   **Iconography:** Use ultra-thin (Light or Regular weight) line icons. Icons should never be boxed; they should float freely next to their associated data point.
*   **Data Vitals:** Large-scale typography (Display-LG) paired with a sparkline. Use scale to indicate importance—vital signs should be 3x larger than their descriptive labels.