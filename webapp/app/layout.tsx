import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Display font, used for the title and accents (body text is a system stack).
const montreuilPlay = localFont({
  src: "./fonts/MontreuilPlayDemo-Regular.otf",
  variable: "--font-montreuil-play",
});

export const metadata: Metadata = {
  title: "Samuel",
  description: "Speak — the vocal tract model mimics you.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${montreuilPlay.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
