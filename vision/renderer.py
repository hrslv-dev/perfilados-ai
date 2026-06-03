import cv2

class Renderer: 

    # Desenhar retângulo 
    def draw_boundering_box( 
            self, 
            frame, 
            x,
            y, 
            w,
            h, 
            color=(255,0,0)
    ): 
        cv2.rectangle(
            frame, 
            (x,y), 
            (x + w, y + h), 
            color, 
            2
        )

    # Renderização textual 
    def draw_text(
            self,
            frame, 
            text, 
            position, 
            color= (0,255,0)
    ): 
        cv2.putText(
            frame,
            text,
            position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            2
        )
    def draw_features(
         self, 
         frame, 
         features
    ): 
        x = features["x"]
        y = features["y"]
        area = features["area"]
        aspect_ratio = features["aspect_ratio"]
        circularity = features["circularity"]

        self.draw_text(
            frame, 
            f"Area: {int(area)}",
            (x,y,-10), 
            (0,255,0)
        )

        self.draw_text(
            frame,
            f"AR: {aspect_ratio:.2f}",
            (x, y - 35),
            (0, 255, 255)
        )

        self.draw_text(
            frame,
            f"Circ: {circularity:.2f}",
            (x, y - 60),
            (255, 255, 0)
        )

    def draw_complete_overlay(
        self,
        frame,
        features
    ):

        x = features["x"]
        y = features["y"]

        w = features["width"]
        h = features["height"]


        self.draw_boundering_box(
            frame,
            x,
            y,
            w,
            h
            
        )

        self.draw_features(
            frame,
            features
        )

        return frame
        