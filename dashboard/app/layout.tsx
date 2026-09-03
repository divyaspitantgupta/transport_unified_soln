import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// Premium sans-serif font load kar rahe hain
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Smart City Pipeline",
  description: "Live Command Centre",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // 'dark' class ensure karegi ki Shadcn black mode mein chale
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-black antialiased`}>
        {children}
      </body>
    </html>
  );
}