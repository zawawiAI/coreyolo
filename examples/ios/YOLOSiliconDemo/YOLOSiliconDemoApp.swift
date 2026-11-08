import SwiftUI

@main
struct YOLOSiliconDemoApp: App {
    var body: some Scene {
        WindowGroup {
            CameraDetectView()
                .statusBarHidden(true)
        }
    }
}
