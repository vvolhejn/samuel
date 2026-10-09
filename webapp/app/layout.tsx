import type { Metadata, Viewport } from "next";
import { Analytics } from "@/components/Analytics";
import "./globals.css";

const DESCRIPTION =
  "Samuel is a model that learns to control a silly mouth to mimic speech. Pink Trombone automated.";

export const metadata: Metadata = {
  // Needed for the OG/Twitter image paths below to resolve to absolute URLs.
  metadataBase: new URL("https://samuel.vvolhejn.com"),
  title: "Samuel",
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    url: "/",
    siteName: "Samuel",
    title: "Samuel",
    description: DESCRIPTION,
    images: [
      {
        url: "/og.png",
        width: 1200,
        height: 630,
        alt: "A cross-section of the Pink Trombone vocal tract, next to the title 'Samuel'.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Samuel",
    description: DESCRIPTION,
    images: ["/og.png"],
  },
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <Analytics />
        {children}
      </body>
    </html>
  );
}
