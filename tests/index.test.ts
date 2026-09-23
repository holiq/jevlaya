import { describe, expect, test } from "bun:test";
import packageJson from "../package.json";

describe("jevlaya project", () => {
  test("package.json metadata", () => {
    expect(packageJson.name).toBe("jevlaya");
    expect(packageJson.packageManager).toBe("bun@1.4.2");
  });
});
